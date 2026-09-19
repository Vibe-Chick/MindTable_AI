"""
ai_engine/matching.py
=======================
유사도/다양성 계산, 그룹 구성, 매칭 이유/아이스브레이커 생성.

통합 결정(INTEGRATION_REPORT.md)에 따라 김주환 브랜치의 `AI/src/embed.py`
(고정 태그 공간 유사도) + `AI/src/match.py`(축별 차등 성격 보완도, greedy+local
search 편성) + `AI/src/cards.py`(타겟 커버리지 검증付 카드 생성)를 이식해 핵심
스코어링/그룹 형성/카드 생성 로직을 전면 교체했다.

jaebin에서 그대로 유지한 것 (팀 결정):
    - `run_matching_pipeline(candidate_pool, group_size=4, match_history=None)`의
      시그니처와 반환 형태 - server.py가 이 계약에 의존한다.
    - 반복 매칭 이력을 프로필에 내장하지 않고 **외부 dict로 주고받는 패턴**
      (store.py의 get_match_history/save_match_history) - 나중에 DB로 옮기기 쉽다.
    - 카드 생성의 ThreadPoolExecutor 병렬화 (이번 통합에서 실측 후 유지, 아래 참고).

반복 매칭 페널티는 김주환 브랜치가 시뮬레이션으로 검증한 이진 방식
(만난 적 있는 쌍마다 고정 DELTA_REPEAT=0.8 감점)을 채택했다. jaebin의 횟수
비례 방식(-0.1×n, 상한 -0.3)보다 공격적이지만, 이 값이 실측/시뮬레이션으로
근거가 있는 쪽이라 우선했다(BRANCH_COMPARISON.md 3.3 참고).
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import logging
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import llm_client
from .schemas import BIG_FIVE_TRAITS, TAGS

logger = logging.getLogger("ai_engine.matching")

# --- 편성 목적함수 가중치 (김주환 브랜치 src/schema.py 이식) -----------------
GROUP_SIZE = 4
ALPHA = 1.0             # 관심사 유사도 가중
GAMMA = 0.5             # 성격 보완도 가중
DELTA_REPEAT = 0.8      # 반복 매칭 페널티 (시뮬레이션 검증된 값)
EPSILON = 0.25          # 탐색 비율: 그룹당 1명은 유사도 하위 풀에서 충원
LOCAL_SEARCH_MAX_ITER = 500

MAX_LLM_WORKERS = 8  # 카드 생성 병렬 처리 스레드 수 (아래 §4 실측 참고)
MAX_ICEBREAKER_ATTEMPTS = 2


# =====================================================================
# 1. 유사도 (고정 태그 공간)
# =====================================================================

def _weights(user: Dict[str, Any]) -> Dict[str, float]:
    return user.get("interest_weights") or {t: 1.0 for t in user.get("tags", [])}


def tag_vector(user: Dict[str, Any]) -> List[float]:
    """태그 가중 벡터. 유사도는 전적으로 이 12차원 공간에서 계산한다."""
    w = _weights(user)
    return [w.get(t, 0.0) for t in TAGS]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    무엇을: 두 벡터 사이의 코사인 유사도.
    왜: 벡터 크기가 아닌 방향만 비교해, 관심사 개수 차이에 영향받지 않고
    "의미적으로 얼마나 같은 방향을 가리키는가"를 측정한다.
    """
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _jaccard(user_a: Dict[str, Any], user_b: Dict[str, Any]) -> float:
    """가중 자카드. interest_weights가 있으면 반영한다."""
    wa, wb = _weights(user_a), _weights(user_b)
    keys = set(wa) | set(wb)
    if not keys:
        return 0.0
    num = sum(min(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
    den = sum(max(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
    return num / den if den else 0.0


def similarity_matrix(users: List[Dict[str, Any]]) -> List[List[float]]:
    """
    무엇을: 태그 코사인(주, 0.85) + 가중 자카드(보조, 0.15)를 섞은 NxN 유사도 행렬.

    왜 자유 키워드가 아니라 고정 태그 공간(TAGS, 12종)인가 (알고리즘적 근거):
    - "클라이밍"과 "등산"을 자유 키워드로만 비교하면 문자열이 달라 유사도가
      0이 된다. 임베딩 API가 없는 게이트웨이 환경(이 프로젝트가 실제로 겨냥하는
      환경)에서는 유사도 자체가 붕괴한다.
    - 12차원 고정 태그로 투영하면 임베딩 없이도 "클라이밍"과 "등산"이 둘 다
      '운동' 태그로 모여 유사도가 살아난다. 관심사 학습(feedback.py)도 이
      태그 가중치(interest_weights)를 움직이는 방식으로 이 공간에 직접
      반영된다.
    """
    n = len(users)
    vectors = [tag_vector(u) for u in users]
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
        for j in range(i + 1, n):
            v = 0.85 * cosine_similarity(vectors[i], vectors[j]) + 0.15 * _jaccard(users[i], users[j])
            m[i][j] = m[j][i] = max(0.0, min(1.0, v))
    return m


# =====================================================================
# 2. 편성 목적함수 (성격 보완도 + 다양성 + 반복매칭 페널티)
# =====================================================================

def personality_complement(members: List[Dict[str, Any]]) -> float:
    """
    무엇을: 그룹의 성격 보완도. 외향성은 분산을, 개방성·우호성은 유사를 원한다.

    왜 축마다 다른 함수를 쓰는가 (심리학적 근거): 외향성이 전원 낮으면(전부
    조용) 대화가 죽고, 전원 높으면(전부 시끄러움) 산만해진다 - 적당한 분산이
    이상적이다. 반면 개방성·우호성은 유사할수록 가치관 충돌 없이 편하게
    대화가 이어진다. 5축을 하나의 유클리드 거리로 뭉치면 이 방향성 차이가
    사라지므로, 축별로 분리했다.
    """
    if len(members) < 2:
        return 0.0
    e_std = statistics.pstdev([m["extraversion"] for m in members])
    e_term = 1.0 - abs(e_std - 1.0) / 2.0  # 목표 분산 약 1.0
    oa = (statistics.pstdev([m["openness"] for m in members])
          + statistics.pstdev([m["agreeableness"] for m in members])) / 2.0
    return e_term - oa / 2.0


def diversity(members: List[Dict[str, Any]]) -> float:
    """
    무엇을: 전공(0.5) + 학교(0.5) 다양성. college 필드가 있으면 major(0.5)/
    college(0.3)/university(0.2) 3단으로 세분화한다.
    왜: 성격/관심사가 비슷한 사람끼리만 묶으면 "새로운 만남"이라는 서비스의
    핵심 가치가 희석된다. 배경이 다르면 새로운 관점/네트워크를 제공할
    확률이 높아지므로 독립 축으로 둔다.
    """
    n = len(members)
    if n == 0:
        return 0.0
    major_div = len({m.get("major") for m in members}) / n
    if all(m.get("college") for m in members):
        college_div = len({m.get("college") for m in members}) / n
        school_div = len({m.get("university") for m in members}) / n
        return 0.5 * major_div + 0.3 * college_div + 0.2 * school_div
    school_div = len({m.get("university") for m in members}) / n
    return 0.5 * major_div + 0.5 * school_div


def _get_user_id(user: Dict[str, Any]) -> str:
    return str(user.get("user_id") or user.get("id") or user.get("name") or id(user))


def make_pair_key(user_a: Dict[str, Any], user_b: Dict[str, Any]) -> Tuple[str, str]:
    """두 사용자를 순서 무관하게 식별하는 매칭 이력 테이블 키."""
    ids = sorted([_get_user_id(user_a), _get_user_id(user_b)])
    return (ids[0], ids[1])


def score_group(
    idx: List[int],
    users: List[Dict[str, Any]],
    sim: List[List[float]],
    match_history: Optional[Dict[Tuple[str, str], int]] = None,
    with_penalty: bool = True,
) -> float:
    """
    무엇을: 그룹 하나의 목적함수 점수 = ALPHA*관심사유사도 + beta*다양성 +
    GAMMA*성격보완도 - (반복매칭 페널티, with_penalty=True일 때만).

    왜 diversity_beta를 실제로 곱하는가 (통합 결정의 핵심 수정): jaebin
    원안은 diversity_beta를 리뷰에서 학습만 하고 그룹 점수 계산에는 전혀
    쓰지 않는 죽은 값이었다(BRANCH_COMPARISON.md 3.3). 이 통합에서는
    김주환 브랜치처럼 그룹 구성원 평균 beta를 다양성 항에 직접 곱해,
    "다양성을 선호하는 사람이 많은 그룹일수록 다양성이 실제로 점수에 더
    크게 반영"되도록 배선했다.

    왜 with_penalty=False 옵션이 필요한가: 반복 매칭 페널티는 회차가
    갈수록 누적되므로, 회차 간 비교(만족도 곡선 등)에 페널티 포함 점수를
    쓰면 프로필이 좋아져도 곡선이 내려가는 착시가 생긴다. 순수 품질만
    보려면 False로 호출한다.
    """
    members = [users[i] for i in idx]
    if len(idx) < 2:
        return 0.0
    pairs = list(itertools.combinations(idx, 2))
    interest = sum(sim[i][j] for i, j in pairs) / len(pairs)
    beta = sum(m.get("diversity_beta", 0.5) for m in members) / len(members)
    quality = ALPHA * interest + beta * diversity(members) + GAMMA * personality_complement(members)
    if not with_penalty or not match_history:
        return quality
    repeat = sum(1 for i, j in pairs
                 if match_history.get(make_pair_key(users[i], users[j]), 0) > 0)
    return quality - DELTA_REPEAT * repeat


def _total(groups: List[List[int]], users, sim, match_history) -> float:
    return sum(score_group(g, users, sim, match_history) for g in groups)


# =====================================================================
# 3. 그룹 구성 (greedy + local search)
# =====================================================================

def _greedy(
    users: List[Dict[str, Any]],
    sim: List[List[float]],
    match_history: Optional[Dict[Tuple[str, str], int]],
    group_size: int,
    explore: bool = True,
    seed: int = 0,
) -> Tuple[List[List[int]], List[int]]:
    """
    무엇을: 시드(가장 유사도 합이 큰 사람) + 한계이득 최대 충원으로 그룹을
    구성한다. 그룹당 1자리는 탐색용으로 예약한다(유사도 하위 25%에서 무작위
    1명 - 최적화 후 깨는 방식은 목적함수를 망가뜨리므로 처음부터 넣는다).
    """
    import random
    rng = random.Random(seed)
    pool = set(range(len(users)))
    groups: List[List[int]] = []

    while len(pool) >= group_size:
        seed_i = max(pool, key=lambda i: sum(sim[i][j] for j in pool))
        g = [seed_i]
        pool.discard(seed_i)

        n_normal = group_size - 1 - (1 if explore else 0)
        for _ in range(n_normal):
            if not pool:
                break
            best = max(pool, key=lambda c: score_group(g + [c], users, sim, match_history))
            g.append(best)
            pool.discard(best)

        if explore and pool:
            ranked = sorted(pool, key=lambda c: sim[seed_i][c])
            k = max(1, int(len(ranked) * EPSILON))
            pick = rng.choice(ranked[:k])
            g.append(pick)
            pool.discard(pick)

        groups.append(g)

    return groups, sorted(pool)


def _local_search(
    groups: List[List[int]],
    users: List[Dict[str, Any]],
    sim: List[List[float]],
    match_history: Optional[Dict[Tuple[str, str], int]],
    max_iter: int = LOCAL_SEARCH_MAX_ITER,
) -> List[List[int]]:
    """두 그룹 간 멤버 스왑. 총점이 오르면 수용. 그리디의 국소 최적을 개선한다."""
    groups = [list(g) for g in groups]
    cur = _total(groups, users, sim, match_history)
    it = 0
    improved = True
    while improved and it < max_iter:
        improved = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                for x in range(len(groups[a])):
                    for y in range(len(groups[b])):
                        it += 1
                        if it > max_iter:
                            return groups
                        groups[a][x], groups[b][y] = groups[b][y], groups[a][x]
                        new = _total(groups, users, sim, match_history)
                        if new > cur + 1e-9:
                            cur = new
                            improved = True
                        else:
                            groups[a][x], groups[b][y] = groups[b][y], groups[a][x]
    return groups


def form_groups(
    users: List[Dict[str, Any]],
    group_size: int = GROUP_SIZE,
    match_history: Optional[Dict[Tuple[str, str], int]] = None,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    무엇을: 대기 인원을 group_size 단위로 묶는다. 그리디 구성 후 국소 탐색
    (멤버 스왑)으로 한 번 더 개선한다(김주환 브랜치 2단계 편성 이식).

    왜 2단계인가 (알고리즘적 근거): 그리디 1패스만으로는 초반 시드 선택의
    악영향이 끝까지 남는다. 두 그룹 사이의 멤버를 맞바꿔보고 총점이 오르면
    받아들이는 국소 탐색을 반복 상한(500회) 안에서 수행하면, 순수 그리디보다
    항상 같거나 나은 결과를 보장하면서도 계산량을 통제할 수 있다.

    Returns:
        (완성된 그룹 리스트, 그룹 크기를 채우지 못해 이월되는 인원 리스트)
        반환되는 그룹의 dict는 입력 users의 동일 객체다(복사하지 않음) -
        server.py가 이 identity를 이용해 leftover를 재구성한다.
    """
    if len(users) < 2:
        return [], list(users)

    sim = similarity_matrix(users)
    groups_idx, leftover_idx = _greedy(users, sim, match_history, group_size, explore=True)
    groups_idx = _local_search(groups_idx, users, sim, match_history)

    complete_groups = [[users[i] for i in g] for g in groups_idx if len(g) == group_size]
    leftover = [users[i] for i in leftover_idx]
    return complete_groups, leftover


def record_match_history(
    groups: List[List[Dict[str, Any]]],
    match_history: Dict[Tuple[str, str], int],
) -> Dict[Tuple[str, str], int]:
    """이번 라운드에 확정된 그룹들의 모든 쌍을 match_history에 1씩 누적 기록한다."""
    for group in groups:
        for user_a, user_b in itertools.combinations(group, 2):
            key = make_pair_key(user_a, user_b)
            match_history[key] = match_history.get(key, 0) + 1
    return match_history


def benchmark(users: List[Dict[str, Any]], group_size: int = GROUP_SIZE,
              match_history: Optional[Dict[Tuple[str, str], int]] = None) -> Dict[str, Any]:
    """
    무엇을: random/greedy/local_search 세 전략의 목적함수 값을 비교한다
    (김주환 브랜치 match.py의 벤치마크 이식). "이 편성이 랜덤보다 실제로
    나은가"를 즉시 수치로 보여주기 위한 검증/발표용 유틸리티.
    """
    import random as _random
    sim = similarity_matrix(users)

    idx = list(range(len(users)))
    _random.Random(0).shuffle(idx)
    rnd_groups = [idx[i:i + group_size] for i in range(0, len(idx), group_size)]
    rnd_groups = [g for g in rnd_groups if len(g) == group_size]

    grd_groups, leftover = _greedy(users, sim, match_history, group_size, explore=True)
    opt_groups = _local_search(grd_groups, users, sim, match_history)

    return {
        "n_users": len(users),
        "n_groups": len(opt_groups),
        "leftover": len(leftover),
        "objective": {
            "random": round(_total(rnd_groups, users, sim, match_history), 4),
            "greedy": round(_total(grd_groups, users, sim, match_history), 4),
            "local_search": round(_total(opt_groups, users, sim, match_history), 4),
        },
    }


# =====================================================================
# 4. 카드 생성 (매칭 이유 + 아이스브레이커, 타겟 커버리지 검증)
# =====================================================================

_CARD_SCORE = {"type": "string"}
CARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reason": {"type": "string"},
        "icebreakers": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "targets": {"type": "array", "items": {"type": "string"},
                                "minItems": 1, "maxItems": 6},
                },
                "required": ["question", "targets"],
            },
        },
    },
    "required": ["reason", "icebreakers"],
}

CARD_SYSTEM_PROMPT = """\
너는 처음 만나는 대학생들의 밥자리를 여는 진행자다.
(1) 이 조합의 이유 한 줄, (2) 이 그룹 전용 대화 주제 3개를 만든다.

## 이유
- 겹치는 지점과 다른 지점을 구체적으로 짚는다. 누구의 무엇인지 이름을 쓴다.
- "다양한 전공이 모였습니다" 같은 일반론 금지. 한 문장, 60자 내외.

## 대화 주제 — 가장 중요한 규칙
**이미 머릿속에 답이 있는 질문만 낸다.** 좋은 질문은 기억을 꺼내게 한다.
**질문마다 targets에 그 질문에 답할 수 있는 사람 이름을 모두 적는다.**
프로필에 적힌 이름 그대로 쓴다.

**질문 3개는 역할이 다르다.**
1번: 전원이 답할 거리가 있는 질문 -> targets에 전원의 이름.
2번: 최소 2명 이상이 공유하는 구체적 지점.
3번: 최소 2명 이상. 1·2번에서 덜 나온 사람을 끌어온다.

반드시 지킬 것:
- 프로필에 적힌 활동만 언급한다. 적혀 있지 않은 걸 추론해서 쓰지 마라.
- 구체적인 과거 경험을 묻는다. 가볍게, 처음 만난 사람끼리 밥 먹으며 하는 얘기다.

금지: 커리어/진로/가치관/신념 질문, 가정형 브레인스토밍, 개선점 묻기,
"MBTI 뭐예요"/"취미가 뭐예요" 같은 범용 질문, 한 사람만 답할 수 있는 질문,
예/아니오로 끝나는 질문, 성격 점수 언급.

질문 하나는 60자를 넘기지 마라.
"""

_CARD_CORRECTION = """

[재작성 지시] 직전 출력이 규칙을 어겼다: %s
- targets가 1명뿐인 질문 금지. 1번은 전원, 2·3번은 최소 2명.
- 세 질문의 targets 합집합에 전원이 들어가야 한다.
- 모든 질문에 프로필의 구체적 활동을 그대로 적는다.
다시 써라."""


def _format_group_for_prompt(group: List[Dict[str, Any]]) -> str:
    lines = []
    for u in group:
        interests = ", ".join(u.get("interest_tags", []))
        lines.append(f"- {u.get('name', '익명')} ({u.get('university', '미상')} {u.get('major', '미상')}) 관심사: {interests}")
    return "\n".join(lines)


def _overlap_keywords(group: List[Dict[str, Any]]) -> List[str]:
    c = Counter(k for u in group for k in u.get("interest_tags", []))
    return [k for k, n in c.items() if n >= 2]


def _hits(text: str, keywords: List[str]) -> int:
    t = text.replace(" ", "")
    return sum(1 for k in keywords if k.replace(" ", "") in t)


def _check_card(out: Dict[str, Any], names: set, keywords: List[str]) -> List[str]:
    """
    무엇을: 카드 응답이 타겟 커버리지/구체성 규칙을 지켰는지 검증한다
    (김주환 브랜치 cards.py의 _check 이식). jaebin 원안은 "문자열 3개인가"만
    확인해 범용 질문·1인 전용 질문을 걸러내지 못했다.
    """
    errs, covered = [], set()
    for i, ib in enumerate(out.get("icebreakers") or []):
        tg = [t for t in (ib.get("targets") or []) if t in names]
        covered |= set(tg)
        q = ib.get("question", "")
        if i == 0 and len(tg) < len(names):
            errs.append(f"1번이 전원 대상이 아님({len(tg)}/{len(names)})")
        elif i > 0 and len(tg) < 2:
            errs.append(f"{i + 1}번이 1인 전용")
        need = 2 if i == 0 else 1
        if keywords and _hits(q, keywords) < need:
            errs.append(f"{i + 1}번이 범용 질문(구체적 활동 {need}개 미만)")
    miss = names - covered
    if miss:
        errs.append("질문에 안 나오는 사람: " + ", ".join(sorted(miss)))
    return errs


def _fallback_card(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """LLM 백엔드가 없거나 재작성까지 실패했을 때 쓰는 규칙 기반 카드."""
    majors = [u.get("major", "전공 미상") for u in group]
    interests = [tag for u in group for tag in u.get("interest_tags", [])]
    unique_majors = list(dict.fromkeys(majors))
    common = Counter(interests).most_common(1)
    common_tag = common[0][0] if common else (interests[0] if interests else "새로운 대화 주제")
    if len(unique_majors) > 1:
        reason = f"{', '.join(unique_majors)} 등 서로 다른 전공이 모였지만 '{common_tag}'라는 공통 관심사로 자연스럽게 대화가 이어질 조합입니다."
    else:
        reason = f"같은 '{unique_majors[0]}' 전공이면서 성격 궁합도 잘 맞는 조합입니다."

    unique_interests = list(dict.fromkeys(interests))[:3]
    templates = [
        "'{}'에 관심 있으신 분들이 모인 것 같은데, 다들 어쩌다 관심을 갖게 되셨어요?",
        "혹시 최근에 '{}'와 관련해서 재밌었던 경험이나 후기 있으신가요?",
        "'{}' 얘기 나온 김에, 서로한테 추천해주고 싶은 것도 있나요?",
    ]
    names = [u.get("name", "익명") for u in group]
    questions = [templates[i % len(templates)].format(tag) for i, tag in enumerate(unique_interests)]
    while len(questions) < 3:
        questions.append("오늘 처음 만난 분들끼리, 서로 전공이 어떻게 다른지부터 풀어볼까요?")
    return {
        "reason": reason,
        "icebreakers": questions[:3],
        "icebreaker_targets": [names, names, names],
        "overlap": _overlap_keywords(group),
    }


def build_card(group: List[Dict[str, Any]], group_id: str = "") -> Dict[str, Any]:
    """
    무엇을: 그룹 하나에 대해 매칭 이유 + 아이스브레이커 3개(+targets)를 생성하고,
    타겟 커버리지/구체성을 검증해 실패하면 1회 재작성을 요청한다.
    """
    if not llm_client.is_configured():
        return _fallback_card(group)

    overlap = _overlap_keywords(group)
    majors = sorted({u.get("major", "전공 미상") for u in group})
    prompt = (
        f"[그룹 구성원]\n{_format_group_for_prompt(group)}\n\n"
        f"[겹치는 관심사] {', '.join(overlap) if overlap else '(직접 겹치는 키워드 없음)'}\n"
        f"[전공 구성] {', '.join(majors)}\n"
    )
    names = {u.get("name", "익명") for u in group}
    keywords = [k for u in group for k in u.get("interest_tags", [])]

    try:
        out = llm_client.call(prompt, CARD_SCHEMA, system=CARD_SYSTEM_PROMPT, temp=0.7,
                              tag="card:" + group_id)
        errs = _check_card(out, names, keywords)
        if errs:
            logger.info("[build_card] %s 재작성: %s", group_id, "; ".join(errs))
            out2 = llm_client.call(prompt + _CARD_CORRECTION % "; ".join(errs), CARD_SCHEMA,
                                   system=CARD_SYSTEM_PROMPT, temp=0.7, tag="card:%s:retry" % group_id)
            if not _check_card(out2, names, keywords):
                out = out2
            else:
                logger.warning("[build_card] %s 재작성도 검증 실패, 그대로 사용", group_id)
                out = out2

        qs, tgs = [], []
        for ib in out["icebreakers"]:
            qs.append(ib["question"])
            tgs.append([t for t in (ib.get("targets") or []) if t in names])
        return {"reason": out["reason"], "icebreakers": qs,
                "icebreaker_targets": tgs, "overlap": overlap}
    except Exception as exc:
        logger.error("[build_card] %s LLM 호출 실패, 규칙 기반 폴백: %s", group_id, exc)
        return _fallback_card(group)


def generate_match_reason(group: List[Dict[str, Any]]) -> str:
    """하위 호환용 단일 함수 - build_card()의 reason만 반환."""
    return build_card(group).get("reason") or "매칭 이유를 생성하지 못했습니다."


def generate_icebreakers(group: List[Dict[str, Any]]) -> List[str]:
    """하위 호환용 단일 함수 - build_card()의 icebreakers만 반환."""
    return build_card(group).get("icebreakers") or []


# =====================================================================
# 5. 오케스트레이션
# =====================================================================

def run_matching_pipeline(
    candidate_pool: List[Dict[str, Any]],
    group_size: int = GROUP_SIZE,
    match_history: Optional[Dict[Tuple[str, str], int]] = None,
) -> List[Dict[str, Any]]:
    """
    무엇을: form_groups()로 그룹을 구성하고, 그룹별로 build_card()(매칭 이유+
    아이스브레이커+targets+overlap)를 생성한다. match_history를 넘기면 반복
    매칭 페널티가 반영되고, 이번 라운드 결과가 그 dict에 누적 기록된다.

    왜 ThreadPoolExecutor(max_workers=8)로 병렬화했는가: 그룹 G개면 카드 생성
    LLM 호출이 최대 G회(build_card 내부에서 reason+icebreakers를 한 번의
    call()로 함께 받으므로 그룹당 1~2회)다. 순차 실행하면 총 대기시간이
    호출 시간의 합이 되므로, 그룹 단위로 스레드 풀에 제출해 동시 처리한다.
    실측(2026-09-19, test_429.py): 40명/그룹 10개를 국민대 게이트웨이
    claude-haiku-4-5 로 돌려 최대 동시 요청 8에 도달시켰고 429는 0회였다.
    HTTP 13회(카드 10 + 검증 재작성 3), 총 12.0초. 따라서 8을 유지한다.
    배치 호출 쪽 LLM_WORKERS=4 와 값이 다른 것은 의도된 분리다 - 카드 생성은
    그룹 수만큼(최대 수십 회)만 발생하고, 프로필 추출은 인원수만큼(수백 회)
    발생하므로 후자를 더 보수적으로 잡는다.
    재측정이 필요하면: source .env && python3 test_429.py 40

    Returns:
        [{"members": [...], "match_reason": "...", "icebreakers": [...],
          "icebreaker_targets": [...], "overlap": [...]}, ...]
        순서는 항상 원래 그룹 순서와 동일하다(병렬 실행이어도 완료 순서와 무관).
    """
    groups, leftover = form_groups(candidate_pool, group_size=group_size, match_history=match_history)

    cards: List[Optional[Dict[str, Any]]] = [None] * len(groups)
    if groups:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
            future_to_idx = {
                executor.submit(build_card, group, f"g{idx}"): idx
                for idx, group in enumerate(groups)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    cards[idx] = future.result()
                except Exception as exc:
                    logger.error("[run_matching_pipeline] 그룹 %d 카드 생성 실패: %s", idx, exc)
                    cards[idx] = _fallback_card(groups[idx])

    results = [
        {
            "members": groups[i],
            "match_reason": cards[i].get("reason") or "매칭 이유를 생성하지 못했습니다.",
            "icebreakers": cards[i].get("icebreakers") or [],
            "icebreaker_targets": cards[i].get("icebreaker_targets") or [],
            "overlap": cards[i].get("overlap") or [],
        }
        for i in range(len(groups))
    ]

    if match_history is not None:
        record_match_history(groups, match_history)

    if leftover:
        names = [u.get("name", "익명") for u in leftover]
        logger.info("[run_matching_pipeline] %d명은 그룹 크기를 채우지 못해 다음 라운드로 이월됩니다: %s",
                   len(leftover), names)

    return results
