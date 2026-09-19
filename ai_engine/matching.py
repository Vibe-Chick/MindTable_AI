"""
ai_engine/matching.py
=======================
유사도/다양성 계산, 그룹 구성, 매칭 이유/아이스브레이커 생성.

matching_engine.py의 get_embedding/cosine_similarity/personality_similarity/diversity_score/
pair_score/form_groups/generate_match_reason/generate_icebreakers/run_matching_pipeline을
동작 변경 없이 그대로 옮겼다(실제 학교 게이트웨이로 이미 검증된 로직).

신규 기능 - 반복 매칭 페널티:
    같은 두 사람이 여러 라운드에 걸쳐 계속 묶이면 "새로운 사람을 만난다"는 서비스의 핵심
    가치가 훼손된다. pair_score/form_groups/run_matching_pipeline에 선택적 match_history
    인자를 추가해, 과거 매칭 횟수만큼 점수를 감점하도록 했다. 모듈에 전역 변수로 두지 않고
    호출자가 소유한 dict를 인자로 주고받는 방식을 택한 이유는, 이 패키지의 기존 설계 원칙
    ("함수는 전역 상태를 갖지 않는다")을 지키기 위함이다 - 백엔드는 이 dict를 DB에서 불러와
    넘기고, 결과를 다시 DB에 저장하면 된다.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import itertools
import json
import logging
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .json_utils import _extract_json_array
from .llm_client import _call_llm, _get_chat_backend
from .schemas import BIG_FIVE_TRAITS

# --- 선택적 외부 패키지 (임베딩 전용) ------------------------------------
try:
    import voyageai  # type: ignore
except ImportError:
    voyageai = None  # type: ignore

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore


logger = logging.getLogger("ai_engine.matching")

MAX_ICEBREAKER_ATTEMPTS = 2  # JSON 배열 파싱 실패 시 재시도(1회 재시도 후 폴백)
MAX_LLM_WORKERS = 8  # run_matching_pipeline의 LLM 호출(매칭 이유/아이스브레이커) 병렬 처리 스레드 수

# 반복 매칭 페널티 기본값 (pair_score의 기본 인자와 동일하게 상수로도 노출)
DEFAULT_REPEAT_PENALTY_PER_MATCH = 0.1
DEFAULT_MAX_REPEAT_PENALTY = 0.3


# =====================================================================
# 1. 임베딩
# =====================================================================

def _get_voyage_client():
    if voyageai is None:
        return None
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        return None
    try:
        return voyageai.Client(api_key=api_key)
    except Exception as exc:  # pragma: no cover
        logger.error(f"Voyage 클라이언트 생성 실패: {exc}")
        return None


def _get_openai_client():
    if openai is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return openai.OpenAI(api_key=api_key)
    except Exception as exc:  # pragma: no cover
        logger.error(f"OpenAI 클라이언트 생성 실패: {exc}")
        return None


def _fallback_embedding(text: str, dim: int = 64) -> List[float]:
    """
    무엇을: 임베딩 API를 쓸 수 없을 때, 텍스트를 해시로 시드 고정한 결정론적 유사-임베딩으로 대체한다.
    왜: 같은 관심사 문자열은 항상 같은 벡터로 매핑되어야 코사인 유사도 비교가 의미를 갖는다.
        진짜 의미 임베딩은 아니지만(단어 유사도 포착 불가), 최소한 "완전히 같은 관심사 조합"과
        "다른 관심사 조합"을 구분해 파이프라인이 끝까지 동작하도록 보장하는 안전망 역할이다.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()


def get_embedding(text: str) -> List[float]:
    """
    무엇을: 관심사 키워드 문자열(예: "등산 보드게임 여행")을 의미 벡터로 변환한다.

    왜 임베딩인가 (알고리즘적 근거):
    - 관심사 유사도를 정량 비교하려면 텍스트를 벡터 공간에 투영해야 한다. 임베딩은
      "등산"과 "캠핑"처럼 표면적 단어는 달라도 의미적으로 가까운 관심사를 포착할 수 있어,
      단순 문자열/키워드 일치보다 실제 대화 궁합을 더 잘 반영한다.
    - 우선순위: Voyage AI(Anthropic 권장 임베딩 파트너) -> OpenAI 임베딩 -> 해시 기반 폴백.
      상위 옵션이 실패하면 조용히 죽는 대신 다음 옵션으로 넘어가 파이프라인을 지속시킨다.
    """
    voyage_client = _get_voyage_client()
    if voyage_client is not None:
        try:
            result = voyage_client.embed([text], model="voyage-3-lite", input_type="document")
            return list(result.embeddings[0])
        except Exception as exc:
            logger.warning(f"[get_embedding] Voyage 임베딩 실패, 다음 옵션으로 폴백: {exc}")

    openai_client = _get_openai_client()
    if openai_client is not None:
        try:
            resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
            return list(resp.data[0].embedding)
        except Exception as exc:
            logger.warning(f"[get_embedding] OpenAI 임베딩 실패, 해시 기반 폴백 사용: {exc}")

    return _fallback_embedding(text)


# =====================================================================
# 2. 유사도 / 다양성 계산
# =====================================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    무엇을: 두 벡터 사이의 코사인 유사도를 순수 numpy로 계산한다(-1~1, 보통 임베딩끼리는 0~1 근방).
    왜: 코사인 유사도는 벡터의 크기(magnitude)가 아닌 방향만 비교하므로, 텍스트 길이 차이 같은
        요인에 영향받지 않고 "의미적으로 얼마나 같은 방향을 가리키는가"만 순수하게 측정할 수 있다.
    """
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def personality_similarity(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> float:
    """
    무엇을: 두 사용자의 Big Five 5축 점수를 5차원 좌표로 보고, 유클리드 거리를 0~1 유사도로
    정규화해 반환한다(거리가 가까울수록 1에 가까움).

    왜 유클리드 거리인가 (심리학적 근거):
    - 유사성-매력 가설(Byrne, 1971, similarity-attraction paradigm)에 따르면 성격 프로파일이
      비슷한 사람들 사이에서 라포 형성과 대화 만족도가 더 높게 나타난다.
    - 5개 축을 하나씩 개별 비교하는 대신 다차원 거리로 묶어서 계산하면, "전체적인 성격 궁합"을
      하나의 값으로 반영할 수 있어 이후 가중합(pair_score)에 넣기 쉽다.
    - 각 축이 1~5 범위이므로 이론상 최대 거리는 sqrt(5 * (5-1)^2) = sqrt(80)이며,
      이를 기준으로 나눠 0~1 범위로 정규화한다.
    """
    vec_a = np.array([profile_a[t] for t in BIG_FIVE_TRAITS], dtype=float)
    vec_b = np.array([profile_b[t] for t in BIG_FIVE_TRAITS], dtype=float)
    distance = float(np.linalg.norm(vec_a - vec_b))
    max_distance = math.sqrt(len(BIG_FIVE_TRAITS) * (5 - 1) ** 2)
    similarity = 1.0 - (distance / max_distance)
    return float(max(0.0, min(1.0, similarity)))


def diversity_score(user_a: Dict[str, Any], user_b: Dict[str, Any]) -> float:
    """
    무엇을: 두 사용자의 major(전공), university(학교)를 비교해 배경 다양성을 0.0/0.5/1.0으로 점수화한다.
      - 전공/학교 둘 다 다르면 1.0
      - 하나만 다르면 0.5
      - 둘 다 같으면 0.0

    왜 배경 다양성을 별도로 측정하는가 (알고리즘적 근거):
    - 성격/관심사가 비슷한 사람끼리만 묶으면 "새로운 만남"이라는 서비스의 핵심 가치가
      희석된다. 전공/학교가 다르면 서로 접하지 못했던 관점과 네트워크를 제공할 확률이
      높아지므로, 이를 독립된 축으로 두어 pair_score에서 성격/관심사 유사도와
      상충(trade-off)하도록 설계했다.
    """
    diff_count = 0
    if user_a.get("major") != user_b.get("major"):
        diff_count += 1
    if user_a.get("university") != user_b.get("university"):
        diff_count += 1
    return diff_count / 2.0


def _get_user_id(user: Dict[str, Any]) -> str:
    """매칭 이력 테이블의 키로 쓸 사용자 식별자. user_id/id가 있으면 우선 쓰고, 없으면 name, 그마저 없으면 객체 identity로 폴백."""
    return str(user.get("user_id") or user.get("id") or user.get("name") or id(user))


def make_pair_key(user_a: Dict[str, Any], user_b: Dict[str, Any]) -> Tuple[str, str]:
    """
    무엇을: 두 사용자를 순서에 무관하게 식별하는 매칭 이력 테이블 키를 만든다
    (정렬된 튜플이라 (A, B)와 (B, A)가 항상 같은 키가 된다).
    """
    ids = sorted([_get_user_id(user_a), _get_user_id(user_b)])
    return (ids[0], ids[1])


def pair_score(
    user_a: Dict[str, Any],
    user_b: Dict[str, Any],
    w_personality: float = 0.4,
    w_interest: float = 0.3,
    w_diversity: float = 0.3,
    match_history: Optional[Dict[Tuple[str, str], int]] = None,
    repeat_penalty_per_match: float = DEFAULT_REPEAT_PENALTY_PER_MATCH,
    max_repeat_penalty: float = DEFAULT_MAX_REPEAT_PENALTY,
) -> float:
    """
    무엇을: personality_similarity, 관심사 임베딩 코사인 유사도, diversity_score를 가중합해
    "이 두 사람을 한 그룹에 묶었을 때 얼마나 좋은 조합인가"를 하나의 스칼라 점수로 만든다.
    match_history가 주어지면, 두 사람이 과거에 몇 번 같은 그룹이었는지에 따라 반복 매칭
    페널티를 추가로 감점한다.

    왜 이 세 요소를 결합하는가 (적정 차별성 이론, Optimal Distinctiveness Theory - Brewer, 1991):
    - 사람은 자신과 너무 비슷한 집단에만 있으면 소속감은 느끼지만 새로움/자극이 없어 지루함을
      느끼고, 반대로 너무 다른 집단에 있으면 소속감 자체를 느끼기 어려워 불편함을 느낀다.
      최적의 만족도는 "동질성(소속 욕구)"과 "이질성(고유성/새로움 욕구)" 사이 균형점에서
      나타난다는 것이 이 이론의 핵심 주장이다.
    - 이를 매칭 알고리즘에 다음과 같이 반영했다:
        * personality_similarity : 성격이 비슷할수록 대화 텐션이 낮고 라포 형성이 쉬움 (동질성 축)
        * 관심사 임베딩 코사인 유사도 : 공통 화제가 있어야 어색함 없이 대화가 이어짐 (동질성 축)
        * diversity_score        : 전공/학교가 다르면 새로운 시각과 네트워크 확장 가치를 제공 (이질성 축)
    - 즉 "성격/관심사는 비슷할수록, 배경(전공/학교)은 다를수록 만족도가 높다"는 가설을
      하나의 점수로 구현한 것이며, 가중치(w_*)는 서비스 운영 중 A/B 테스트로 조정 가능하도록
      함수 인자로 노출해 두었다.

    왜 반복 매칭 페널티가 필요한가 (알고리즘적 근거):
    - 같은 두 사람을 여러 라운드에 걸쳐 계속 묶으면 "새로운 사람을 만난다"는 서비스의
      핵심 가치가 훼손되고, 참가자 풀이 한정된 소수 그룹으로 고착될 위험이 있다.
    - match_history에 기록된 과거 매칭 횟수만큼 점수를 감점해(1회당 -0.1) 이미 여러 번
      만난 조합은 자연스럽게 우선순위에서 밀려나게 한다. 감점 폭에 상한(-0.3)을 둔 이유는,
      아무리 여러 번 만났어도 성격/관심사 궁합이 매우 좋다면 완전히 배제하지는 않기
      위함이다 - "금지"가 아니라 "약한 우선순위 조정"으로 기능하도록 설계했다.

    비고:
    - 관심사 임베딩은 사용자 dict에 "_interest_embedding" 키로 캐싱해, 동일 사용자에 대해
      form_groups()가 반복적으로 pair_score를 호출하더라도 임베딩 API를 중복 호출하지 않는다
      (모듈 전역 상태가 아니라 호출자가 넘긴 dict 자체에만 캐싱하므로 함수의 독립성은 유지된다).
    """
    p_sim = personality_similarity(user_a, user_b)

    # 관심사 태그 순서에 관계없이 같은 벡터가 나오도록 정렬 후 결합.
    tags_a_text = " ".join(sorted(user_a.get("interest_tags", [])))
    tags_b_text = " ".join(sorted(user_b.get("interest_tags", [])))

    emb_a = user_a.get("_interest_embedding")
    if emb_a is None:
        emb_a = get_embedding(tags_a_text)
        user_a["_interest_embedding"] = emb_a

    emb_b = user_b.get("_interest_embedding")
    if emb_b is None:
        emb_b = get_embedding(tags_b_text)
        user_b["_interest_embedding"] = emb_b

    interest_sim = cosine_similarity(emb_a, emb_b)
    interest_sim = max(0.0, interest_sim)  # 음수 유사도가 가중합을 깎아먹지 않도록 0으로 clip

    diversity = diversity_score(user_a, user_b)

    score = w_personality * p_sim + w_interest * interest_sim + w_diversity * diversity

    if match_history:
        repeat_count = match_history.get(make_pair_key(user_a, user_b), 0)
        penalty = min(repeat_count * repeat_penalty_per_match, max_repeat_penalty)
        score -= penalty

    return float(score)


# =====================================================================
# 3. 그룹 구성
# =====================================================================

def form_groups(
    users: List[Dict[str, Any]],
    group_size: int = 4,
    match_history: Optional[Dict[Tuple[str, str], int]] = None,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    무엇을: 대기 인원 전체를 group_size 단위로 묶는다. 모든 쌍의 pair_score를 계산한 뒤,
    점수가 높은 쌍부터 우선적으로 병합하는 그리디 계층적 응집 군집화
    (greedy hierarchical agglomerative clustering, 크기 제약 포함) 방식을 사용한다.
    match_history를 넘기면 각 쌍 점수 계산에 반복 매칭 페널티가 반영된다.

    왜 그리디인가 (알고리즘적 근거):
    - "그룹 내 모든 쌍 점수의 합이 최대가 되도록 인원을 그룹으로 분할"하는 문제는 조합
      최적화 문제로, 인원 수가 늘어나면 완전 탐색이 비현실적이다(경우의 수 폭증).
      실시간 매칭 서비스에서는 매 라운드 마감 시각에 즉시 결과를 내야 하므로,
      "가장 점수가 높은 쌍부터 우선 묶는다"는 그리디 휴리스틱으로 빠르게 준수한 근사해를
      구하는 쪽을 택했다.
    - 두 클러스터를 합쳤을 때 group_size를 넘으면 병합을 금지해, 항상 유효한 그룹 크기
      이하를 유지한다.

    Returns:
        (완성된 그룹 리스트, 그룹 크기를 채우지 못해 다음 라운드로 이월되는 인원 리스트)
    """
    if len(users) < 2:
        return [], list(users)

    # clusters[i] = 클러스터에 속한 사용자 리스트(병합되어 사라진 클러스터는 None으로 표시)
    clusters: List[Optional[List[Dict[str, Any]]]] = [[u] for u in users]
    user_to_cluster: Dict[int, int] = {id(u): i for i, u in enumerate(users)}

    pair_scores: List[Tuple[float, int, int]] = []
    for i, j in itertools.combinations(range(len(users)), 2):
        score = pair_score(users[i], users[j], match_history=match_history)
        pair_scores.append((score, i, j))
    pair_scores.sort(key=lambda x: x[0], reverse=True)

    for score, i, j in pair_scores:
        ci = user_to_cluster[id(users[i])]
        cj = user_to_cluster[id(users[j])]
        if ci == cj:
            continue  # 이미 같은 그룹에 속함
        cluster_i = clusters[ci]
        cluster_j = clusters[cj]
        if cluster_i is None or cluster_j is None:
            continue
        if len(cluster_i) + len(cluster_j) > group_size:
            continue  # 그룹 크기 초과 시 병합 금지

        cluster_i.extend(cluster_j)
        for member in cluster_j:
            user_to_cluster[id(member)] = ci
        clusters[cj] = None

    final_clusters = [c for c in clusters if c]
    complete_groups = [g for g in final_clusters if len(g) == group_size]
    leftover = [member for g in final_clusters if len(g) != group_size for member in g]

    return complete_groups, leftover


def record_match_history(
    groups: List[List[Dict[str, Any]]],
    match_history: Dict[Tuple[str, str], int],
) -> Dict[Tuple[str, str], int]:
    """
    무엇을: 이번 라운드에 확정된 그룹들을 보고, 그룹 내 모든 사용자 쌍의 매칭 횟수를 1씩
    증가시켜 match_history를 갱신한다(같은 dict를 제자리에서 수정하고 반환한다).

    왜: pair_score의 반복 매칭 페널티가 실제로 동작하려면 "누가 누구와 몇 번 만났는지"가
    라운드를 거듭하며 누적되어야 한다. 이 누적을 모듈 전역 변수로 두지 않고 호출자가
    소유한 딕셔너리를 인자로 받아 갱신하는 방식으로 설계해, run_matching_pipeline이
    여러 백엔드 요청에 걸쳐 재사용되어도 요청 간 상태가 뒤섞이지 않는다(각 요청이 자신의
    match_history를 DB 등에서 로드해 넘기고, 갱신된 결과를 다시 저장하면 됨).
    """
    for group in groups:
        for user_a, user_b in itertools.combinations(group, 2):
            key = make_pair_key(user_a, user_b)
            match_history[key] = match_history.get(key, 0) + 1
    return match_history


# =====================================================================
# 4. 매칭 설명 / 아이스브레이커 생성
# =====================================================================

def _format_group_for_prompt(group: List[Dict[str, Any]]) -> str:
    lines = []
    for u in group:
        interests = ", ".join(u.get("interest_tags", []))
        lines.append(f"- {u.get('name', '익명')}: 학교={u.get('university', '미상')}, 전공={u.get('major', '미상')}, 관심사=[{interests}]")
    return "\n".join(lines)


def _fallback_match_reason(group: List[Dict[str, Any]]) -> str:
    """API 키가 없거나 호출이 실패했을 때 사용하는 규칙 기반 매칭 이유 생성."""
    majors = [u.get("major", "전공 미상") for u in group]
    interests = [tag for u in group for tag in u.get("interest_tags", [])]
    unique_majors = list(dict.fromkeys(majors))
    most_common = Counter(interests).most_common(1)
    common_tag = most_common[0][0] if most_common else (interests[0] if interests else "새로운 대화 주제")

    if len(unique_majors) > 1:
        return f"{', '.join(unique_majors)} 등 서로 다른 전공이 모였지만 '{common_tag}'라는 공통 관심사로 자연스럽게 대화가 이어질 조합입니다."
    return f"같은 '{unique_majors[0]}' 전공이면서 성격 궁합도 잘 맞아, 편안하게 '{common_tag}' 이야기부터 풀어갈 수 있는 조합입니다."


def generate_match_reason(group: List[Dict[str, Any]]) -> str:
    """
    무엇을: Claude API를 호출해 그룹원들의 전공/관심사를 근거로 "왜 이 조합인지"를 설명하는
    한 줄짜리 자연어 문장을 생성한다.

    왜: pair_score는 숫자일 뿐이라 사용자가 매칭 결과를 신뢰하고 이해하려면 설명 가능성
    (explainability)이 필요하다. LLM에게 실제 프로필 데이터를 근거로 제시하도록 프롬프트를
    구성해, "적정 차별성 이론"에서 말하는 동질성(관심사)과 이질성(전공)의 균형을 사람이
    읽기 쉬운 언어로 번역한다.
    """
    if _get_chat_backend() is None:
        return _fallback_match_reason(group)

    prompt = (
        "다음은 랜덤 식사 매칭으로 묶인 그룹 구성원 정보야.\n\n"
        f"{_format_group_for_prompt(group)}\n\n"
        "이 조합이 왜 좋은 매칭인지, 전공/학교의 다양성과 공통 관심사를 구체적으로 근거로 삼아 "
        "자연스러운 한국어 한 문장으로 설명해줘. 설명 문장 외의 다른 말은 하지 마."
    )
    try:
        text = _call_llm(prompt, max_tokens=200)
        if not text:
            raise ValueError("빈 응답")
        return text
    except Exception as exc:
        logger.error(f"[generate_match_reason] LLM 호출 실패, 규칙 기반 폴백 사용: {exc}")
        return _fallback_match_reason(group)


def _fallback_icebreakers(group: List[Dict[str, Any]]) -> List[str]:
    """API 키가 없거나 호출이 실패했을 때 사용하는 규칙 기반 아이스브레이커 생성."""
    interests = [tag for u in group for tag in u.get("interest_tags", [])]
    unique_interests = list(dict.fromkeys(interests))[:3]
    templates = [
        "'{}'에 관심 있으신 분들이 모인 것 같은데, 다들 어쩌다 관심을 갖게 되셨어요?",
        "혹시 최근에 '{}'와 관련해서 재밌었던 경험이나 후기 있으신가요?",
        "'{}' 얘기 나온 김에, 서로한테 추천해주고 싶은 것도 있나요?",
    ]
    questions = [templates[i % len(templates)].format(tag) for i, tag in enumerate(unique_interests)]
    while len(questions) < 3:
        questions.append("오늘 처음 만난 분들끼리, 서로 전공이 어떻게 다른지부터 풀어볼까요?")
    return questions[:3]


def generate_icebreakers(group: List[Dict[str, Any]]) -> List[str]:
    """
    무엇을: Claude API를 호출해 이 그룹의 실제 프로필(전공/관심사)에 맞춘 대화 주제 3개를 생성한다.

    왜 범용 질문을 금지하는가: "취미가 뭐예요?" 류의 범용 아이스브레이커는 이미 인터넷에
    흔하고, 그룹의 특성을 전혀 반영하지 못해 어색함 해소 효과가 낮다. 그룹의 실제 관심사/
    전공 키워드를 직접 언급하도록 프롬프트에 명시하면, 참가자가 "이 매칭이 나를 이해하고
    만들어졌다"고 느끼게 되어(개인화 효과) 초반 대화 진입 장벽을 낮춘다.
    """
    if _get_chat_backend() is None:
        return _fallback_icebreakers(group)

    prompt = (
        "다음은 랜덤 식사 매칭 그룹 구성원 정보야.\n\n"
        f"{_format_group_for_prompt(group)}\n\n"
        "이 그룹의 실제 전공과 관심사를 반영해서, 서로 처음 만난 사람들끼리도 자연스럽게 대화를 "
        "시작할 수 있는 아이스브레이킹 질문을 3개 만들어줘. '취미가 뭐예요?' 같은 범용 질문은 안 되고, "
        "위에 나온 구체적인 관심사나 전공 키워드를 반드시 직접 언급해야 해.\n\n"
        "다른 설명, 인사말, 마크다운 없이 아래 형식의 JSON 배열 하나만 출력해. "
        '첫 글자는 반드시 [ 로 시작하고 마지막 글자는 반드시 ] 로 끝나야 해: ["질문1", "질문2", "질문3"]'
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_ICEBREAKER_ATTEMPTS + 1):
        try:
            # max_tokens=400은 한국어 응답 + 게이트웨이 모델 특성상 중간에 잘리기 쉬워 800으로 상향.
            raw_text = _call_llm(prompt, max_tokens=800)
            json_text = _extract_json_array(raw_text)
            questions = json.loads(json_text)
            if not isinstance(questions, list) or len(questions) < 3 or not all(isinstance(q, str) for q in questions):
                raise ValueError(f"아이스브레이커 응답 형식이 올바르지 않음: {questions!r}")
            return [str(q) for q in questions[:3]]
        except Exception as exc:
            last_error = exc
            logger.warning(f"[generate_icebreakers] 시도 {attempt}/{MAX_ICEBREAKER_ATTEMPTS} 실패: {exc}")

    logger.error(f"[generate_icebreakers] 재시도 초과, 규칙 기반 폴백 사용. 마지막 오류: {last_error}")
    return _fallback_icebreakers(group)


# =====================================================================
# 5. 오케스트레이션
# =====================================================================

def run_matching_pipeline(
    candidate_pool: List[Dict[str, Any]],
    group_size: int = 4,
    match_history: Optional[Dict[Tuple[str, str], int]] = None,
) -> List[Dict[str, Any]]:
    """
    무엇을: extract_profile()로 이미 성격/관심사 프로필이 채워진 사용자 목록(candidate_pool)을
    받아, 그룹 구성 -> 매칭 이유 생성 -> 아이스브레이커 생성까지 전체 파이프라인을 실행한다.
    match_history를 넘기면 반복 매칭 페널티가 반영되고, 이번 라운드 결과가 그 dict에
    누적 기록된다(호출자가 다음 라운드 호출 시 같은 dict를 다시 넘기면 됨).

    왜 오케스트레이션 함수를 분리했는가: 백엔드 API(POST /match/run) 핸들러는 이 함수 하나만
    호출하면 되도록 하여, 매칭 로직의 세부 구현(그리디 군집화, 프롬프트 구성 등)이 API 계층에
    새어나가지 않게 한다. 각 그룹 처리 중 일부가 실패해도(LLM 호출 오류 등) 다른 그룹 처리에
    영향이 가지 않도록 그룹 단위로 예외를 격리한다.

    왜 ThreadPoolExecutor로 병렬화했는가 (알고리즘적 근거):
    - generate_match_reason과 generate_icebreakers는 서로 의존관계가 없고(둘 다 group만
      읽고, 서로의 결과를 쓰지 않음), 그룹별 처리도 서로 완전히 독립적이다. 즉 그룹이 G개면
      LLM 호출이 최대 2G개 있는데, 이게 전부 순차 실행되면 총 대기 시간이 2G번의 네트워크
      왕복 시간을 그대로 더한 값이 된다.
    - 이 호출들은 CPU 연산이 아니라 네트워크 I/O(HTTP 요청 후 응답 대기)가 대부분이므로,
      파이썬의 GIL이 있어도 스레드 풀로 동시에 띄우면 총 소요 시간이 "가장 느린 호출 하나"에
      가까워진다(이상적으로는 max(개별 호출 시간)에 수렴, 순차 실행의 sum(개별 호출 시간)
      대비 그룹 수/2에 비례해 빨라짐).
    - 그룹별 처리와 그룹 내부의 두 호출을 별도 ThreadPoolExecutor로 중첩하지 않고, 하나의
      풀(max_workers=8)에 "그룹별 match_reason 작업"과 "그룹별 icebreakers 작업"을 전부
      평면적으로 제출하는 방식을 택했다 - 중첩 풀을 쓰면 실제 동시 실행 스레드 수가
      max_workers를 넘어설 수 있어 "동시 실행 상한 8"이라는 의도가 깨지기 때문이다.
    - future.result()에서 발생하는 예외는 기존과 동일하게 그룹 단위로 개별 처리해, 한 그룹의
      LLM 호출 실패가 다른 그룹이나 전체 파이프라인에 영향을 주지 않는다(기존 폴백 문구/빈
      리스트 그대로 유지).
    - 결과 리스트는 그룹 인덱스를 키로 미리 만들어둔 슬롯에 채워 넣으므로, futures가
      완료되는 순서(as_completed)가 뒤섞여도 최종 반환 순서는 항상 원래 groups 순서와
      동일하다.

    Returns:
        [{"members": [...], "match_reason": "...", "icebreakers": [...]}, ...]
        그룹 크기를 채우지 못해 이월된 인원은 반환값에 포함되지 않고 로그로만 남긴다.
    """
    groups, leftover = form_groups(candidate_pool, group_size=group_size, match_history=match_history)

    # 그룹 인덱스 -> {"match_reason": ..., "icebreakers": ...} 슬롯. futures가 어떤 순서로
    # 끝나든 이 슬롯에 idx로 꽂아 넣으므로 최종 결과 순서는 항상 groups와 동일하게 유지된다.
    slots: List[Dict[str, Any]] = [{} for _ in groups]

    if groups:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
            future_to_task: Dict[concurrent.futures.Future, Tuple[int, str]] = {}
            for idx, group in enumerate(groups):
                future_to_task[executor.submit(generate_match_reason, group)] = (idx, "match_reason")
                future_to_task[executor.submit(generate_icebreakers, group)] = (idx, "icebreakers")

            for future in concurrent.futures.as_completed(future_to_task):
                idx, task = future_to_task[future]
                if task == "match_reason":
                    try:
                        slots[idx]["match_reason"] = future.result()
                    except Exception as exc:  # generate_match_reason 자체가 이미 내부 폴백을 갖지만 이중 방어
                        logger.error(f"[run_matching_pipeline] 매칭 이유 생성 중 예기치 못한 오류: {exc}")
                        slots[idx]["match_reason"] = "매칭 이유를 생성하지 못했습니다."
                else:
                    try:
                        slots[idx]["icebreakers"] = future.result()
                    except Exception as exc:
                        logger.error(f"[run_matching_pipeline] 아이스브레이커 생성 중 예기치 못한 오류: {exc}")
                        slots[idx]["icebreakers"] = []

    results: List[Dict[str, Any]] = [
        {"members": groups[idx], "match_reason": slots[idx]["match_reason"], "icebreakers": slots[idx]["icebreakers"]}
        for idx in range(len(groups))
    ]

    if match_history is not None:
        record_match_history(groups, match_history)

    if leftover:
        names = [u.get("name", "익명") for u in leftover]
        logger.info(f"[run_matching_pipeline] {len(leftover)}명은 그룹 크기를 채우지 못해 다음 라운드로 이월됩니다: {names}")

    return results
