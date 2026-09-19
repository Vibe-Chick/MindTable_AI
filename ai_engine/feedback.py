"""
ai_engine/feedback.py
=======================
리뷰 기반 프로필 보정 파이프라인.

통합 결정(INTEGRATION_REPORT.md)에 따라 김주환 브랜치의 `AI/src/review.py`를
이식했다. jaebin 원안 대비 두 가지가 핵심적으로 다르다:

1) EMA(지수이동평균) 감쇠 기반 보정. jaebin 원안은 "누적 + 상한 클립"이라 한 번
   상한 근처까지 가면 반대 방향 신호가 없는 한 계속 그 근처에 머물렀다(일시적
   노이즈가 영구 편향으로 남을 위험). 김주환 브랜치는 매 회차 기존 값을 10%씩
   감쇠시킨 뒤 새 신호를 더해, 일관된 신호만 누적되고 1회성 잡음은 자연히
   사라지게 한다. 이 감쇠율은 숨은 정답 시뮬레이션(AI/src/tune.py 그리드서치)으로
   검증된 값이다.
2) **관심사 가중치 학습 루프**. jaebin 원안은 리뷰가 Big Five와 diversity_beta만
   갱신하고 interest_tags/관심사 유사도 계산 공간은 온보딩 이후 한 번도 갱신하지
   않았다(BRANCH_COMPARISON.md 3.6에서 지적된 핵심 결함). 이번 이식으로 리뷰에서
   "실제로 대화가 통한 주제/죽은 주제"를 추출해 profile["interest_weights"]를
   직접 갱신하고, 이 값이 matching.py의 유사도 계산에 곧바로 반영된다 - 매칭이
   참조하는 유사도 공간이 회차를 거듭할수록 실제 경험을 반영하도록 닫힌 루프를
   완성한다.

개인정보 최소 보관 원칙(jaebin 원안 유지): 리뷰 원문(자유서술 텍스트)은
extract_delta 호출 한 번에만 쓰이고, apply_delta 처리가 끝나면 폐기된다.
프로필에 영구히 남는 것은 원문에서 파생된 숫자(delta, 가중치)뿐이다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import llm_client
from .schemas import BIG_FIVE_TRAITS, TAGS

logger = logging.getLogger("ai_engine.feedback")

REVIEW_TIMEOUT_HOURS = 24.0
MAX_REVIEW_FAILURES = 2

# --- 보정 루프 튜닝 상수 (김주환 브랜치 AI/src/schema.py 그리드서치 결과 이식) ---
LR = 0.5                # 델타 학습률
BEHAVIOR_CLIP = 1.5     # 누적 보정 상한
BEHAVIOR_DECAY = 0.10   # 회차마다 기존 보정을 이만큼 감쇠시킨다
DIVERGENCE_THRESHOLD = 1.0

BETA_STEP = 0.15
BETA_MIN, BETA_MAX = 0.1, 0.9

INTEREST_UP = 0.45      # 대화가 터진 주제 가중 상승
INTEREST_DOWN = 0.30    # 죽은 주제 가중 하락
INTEREST_DECAY = 0.08   # 회차마다 1.0 쪽으로 수축
INTEREST_NEW = 0.25     # 새 태그 초기 가중
INTEREST_MIN, INTEREST_MAX = 0.0, 2.0
INTEREST_DROP = 0.05    # 이 미만이면 목록에서 제거

Q4_MORE_SIMILAR = "더 비슷"
Q4_SAME = "지금 정도"
Q4_MORE_DIFFERENT = "더 달라도 됨"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# --- 델타 스키마/프롬프트 ----------------------------------------------------

_DELTA_STEP = {"type": "number", "enum": [-1, -0.5, 0, 0.5, 1]}
_TOPIC_LIST = {"type": "array", "items": {"type": "string", "enum": list(TAGS)},
               "minItems": 0, "maxItems": 3}

DELTA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "openness": _DELTA_STEP, "conscientiousness": _DELTA_STEP,
        "extraversion": _DELTA_STEP, "agreeableness": _DELTA_STEP,
        "neuroticism": _DELTA_STEP,
        "worked_topics": _TOPIC_LIST,
        "dead_topics": _TOPIC_LIST,
        "evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["openness", "conscientiousness", "extraversion", "agreeableness",
                 "neuroticism", "worked_topics", "dead_topics", "evidence", "confidence"],
}

REVIEW_SYSTEM_PROMPT = """\
너는 식사 모임 후기를 읽고 참가자 '본인'의 프로필 보정값을 산출한다.

중요: 상대방을 평가하지 않는다. 후기를 쓴 사람 본인의 기존 프로필이 실제
행동과 얼마나 어긋났는지만 판단한다.

보정값 기준:
- 0이 기본값이다. 후기에 근거가 없으면 0을 준다.
- 기존 점수와 같은 방향의 행동이 확인되면 0 (이미 맞으므로 고칠 필요 없음).
- 기존 점수와 다른 방향의 행동이 확인될 때만 그 방향으로 보정한다.
- **기존 점수와 정면으로 어긋나는 행동이 후기에 적혀 있으면 ±1을 준다.**
  예: 외향성 2점인 사람이 "제가 말을 제일 많이 했어요" -> extraversion +1
  예: 외향성 5점인 사람이 "거의 듣기만 했어요" -> extraversion -1
- ±0.5는 방향은 보이지만 약한 경우에만.
- evidence에 근거가 된 후기 원문을 인용한다.

confidence 척도 (이 값이 보정 크기를 그대로 곱한다):
- 0.8~1.0: 후기에서 그대로 인용할 수 있는 행동 서술이 있다
- 0.4~0.7: 암시는 되지만 해석이 필요하다
- 0.0~0.3: "재밌었어요" 수준이라 읽을 게 거의 없다

worked_topics/dead_topics:
- worked_topics: 실제로 대화가 터진 주제를 정해진 태그 목록에서 최대 3개.
  자리에서 실제로 오간 얘기만. 프로필의 기존 관심사를 베끼지 마라.
- dead_topics: 말이 끊겼다고 언급된 주제를 태그로. 없으면 빈 배열.
- 근거가 없으면 빈 배열을 준다. 추측해서 채우지 마라.

반드시 위 스키마의 JSON으로만 응답하라.
"""

REVIEW_USER_TEMPLATE = """\
[기존 프로필] 개방성 {openness} / 성실성 {conscientiousness} / 외향성 {extraversion} / \
우호성 {agreeableness} / 신경성 {neuroticism}

[r1] 대화가 제일 잘 풀린 순간과 그때 무슨 얘기 중이었는지
{r1}

[r2] 말이 끊기거나 어색했던 주제
{r2}

[r3] 본인은 주로 이끄는 쪽이었는지 듣는 쪽이었는지
{r3}
"""


def parse_review(raw_answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    무엇을: 리뷰 원본 답변(자유서술 3개 r1~r3 + 강제선택 1개 r4_choice)을 자유서술
    텍스트 리스트와 강제선택 답변으로 분리한다.
    왜: extract_delta는 자유서술 텍스트만 필요하고, 다양성 β 갱신은 강제선택
    답변만 필요하다.
    """
    free_text = [str(raw_answers.get("r1", "")), str(raw_answers.get("r2", "")),
                str(raw_answers.get("r3", ""))]
    forced_choice = str(raw_answers.get("r4_choice", ""))
    return {"free_text": free_text, "forced_choice": forced_choice}


def _effective(self_report: Dict[str, float], behavior: Dict[str, float]) -> Dict[str, float]:
    """실제 매칭에 쓰이는 벡터 = 자기보고 + 행동 보정 (1~5로 clamp)."""
    return {a: _clamp(self_report.get(a, 3.0) + behavior.get(a, 0.0), 1, 5) for a in BIG_FIVE_TRAITS}


def extract_delta(review_answers: Dict[str, Any], self_report_vector: Dict[str, Any],
                  behavior_corrected_vector: Dict[str, Any]) -> Dict[str, Any]:
    """
    무엇을: 리뷰 자유서술 답변 + 현재 유효 벡터(자기보고+행동보정)를 LLM에 전달해
    Big Five 5축 델타(이산 5단계: -1/-0.5/0/0.5/1) + worked_topics/dead_topics +
    evidence + confidence를 구조화된 JSON으로 추출한다.

    왜 이산 델타인가: jaebin 원안의 연속값(-1~1) 대신 김주환 브랜치의 5단계
    이산값을 채택했다 - 모델이 애매한 실수값을 내놓는 것보다 "방향 있음/강하게
    있음/없음"의 명확한 단계로 판단하게 하는 편이 재현성이 높다(시뮬레이션
    검증 근거는 schema.py 튜닝 상수 주석 참고).
    """
    if not llm_client.is_configured():
        return {t: 0.0 for t in BIG_FIVE_TRAITS} | {
            "worked_topics": [], "dead_topics": [], "evidence": "", "confidence": 0.0}

    eff = _effective(self_report_vector, behavior_corrected_vector or self_report_vector)
    prompt = REVIEW_USER_TEMPLATE.format(
        openness=round(eff["openness"], 1), conscientiousness=round(eff["conscientiousness"], 1),
        extraversion=round(eff["extraversion"], 1), agreeableness=round(eff["agreeableness"], 1),
        neuroticism=round(eff["neuroticism"], 1),
        r1=review_answers.get("free_text", ["", "", ""])[0].strip() or "(무응답)",
        r2=review_answers.get("free_text", ["", "", ""])[1].strip() or "(무응답)",
        r3=review_answers.get("free_text", ["", "", ""])[2].strip() or "(무응답)",
    )
    return llm_client.call(prompt, DELTA_SCHEMA, system=REVIEW_SYSTEM_PROMPT, temp=0.0,
                           tag="review")


def apply_delta(profile: Dict[str, Any], delta: Dict[str, Any], q4: Optional[str] = None) -> Dict[str, Any]:
    """
    무엇을: 델타를 EMA(지수이동평균) 방식으로 behavior_corrected_vector에 반영하고,
    worked/dead topics로 interest_weights를 갱신하고, q4 강제선택으로
    diversity_beta를 조정한다. self_report_vector는 절대 건드리지 않는다.

        behavior <- behavior*(1-BEHAVIOR_DECAY) + LR*confidence*delta

    왜 감쇠가 핵심인가: 감쇠 없는 단순 누적은 리뷰 잡음이 랜덤워크로 쌓여
    자기보고가 이미 정확했던 사람의 프로필까지 망가뜨린다(김주환 브랜치
    시뮬레이션으로 확인된 실패 모드). 감쇠가 있으면 일관된 신호만 살아남는다.
    confidence 가중도 같은 이유다 - "재밌었어요"뿐인 후기는 프로필을 거의
    못 움직인다.

    Returns:
        갱신된 profile dict (self_report_vector는 원본과 동일한 값의 새 dict,
        behavior_corrected_vector/diversity_beta/interest_weights/tags만 변경).
    """
    updated = dict(profile)
    self_report = dict(profile["self_report_vector"])
    behavior = dict(profile.get("behavior_corrected_vector") or self_report)

    conf = _clamp(float(delta.get("confidence", 1.0) or 0.0), 0.0, 1.0)
    applied: Dict[str, float] = {}
    for trait in BIG_FIVE_TRAITS:
        d = float(delta.get(trait, 0.0) or 0.0)
        before = behavior.get(trait, 0.0)
        after = _clamp(before * (1.0 - BEHAVIOR_DECAY) + LR * conf * d, -BEHAVIOR_CLIP, BEHAVIOR_CLIP)
        behavior[trait] = after
        applied[trait] = after - before

    updated["self_report_vector"] = self_report
    updated["behavior_corrected_vector"] = behavior

    updated = _apply_topics(updated, delta.get("worked_topics") or [], delta.get("dead_topics") or [], conf)

    beta = float(updated.get("diversity_beta", 0.5))
    if q4 == Q4_MORE_SIMILAR:
        beta = _clamp(beta - BETA_STEP, BETA_MIN, BETA_MAX)
    elif q4 == Q4_MORE_DIFFERENT:
        beta = _clamp(beta + BETA_STEP, BETA_MIN, BETA_MAX)
    updated["diversity_beta"] = beta

    updated["_last_deltas"] = applied  # run_feedback_pipeline이 응답에 실어 보낼 실제 반영량
    return updated


def _apply_topics(profile: Dict[str, Any], worked: List[str], dead: List[str], conf: float = 1.0) -> Dict[str, Any]:
    """
    무엇을: 리뷰에서 나온 "실제로 통한 주제/죽은 주제"로 profile["interest_weights"]를
    갱신한다 - matching.py의 유사도 계산이 참조하는 그 공간이다.

    왜: 설문에 쓴 관심사는 "자기가 생각하는 나"이고, 리뷰의 worked_topics는
    "실제 자리에서 일어난 일"이다. 이 둘이 갈리는 게 자기보고 편향의 가장
    눈에 보이는 형태이며, 이 함수가 matching.py에 실제로 영향을 주는
    유일한 경로다(BRANCH_COMPARISON.md 3.6에서 jaebin 원안에 없던 것으로
    지적된 핵심 기능).

    왜 1.0 쪽으로 수축(INTEREST_DECAY)하는가: 없으면 한 번 언급된 주제가
    영구히 프로필을 지배한다.
    """
    updated = dict(profile)
    w = dict(updated.get("interest_weights") or {t: 1.0 for t in updated.get("tags", [])})

    for k in list(w):
        w[k] += (1.0 - w[k]) * INTEREST_DECAY

    for t in worked:
        t = (t or "").strip()
        if not t:
            continue
        cur = w.get(t, INTEREST_NEW)
        w[t] = _clamp(cur + INTEREST_UP * conf, INTEREST_MIN, INTEREST_MAX)

    for t in dead:
        t = (t or "").strip()
        if t in w:
            w[t] = _clamp(w[t] - INTEREST_DOWN * conf, INTEREST_MIN, INTEREST_MAX)

    for k in [k for k, v in w.items() if v < INTEREST_DROP]:
        del w[k]

    updated["interest_weights"] = w
    updated["tags"] = [k for k, _ in sorted(w.items(), key=lambda kv: -kv[1])]
    return updated


def update_diversity_beta(profile: Dict[str, Any], forced_choice_answer: str) -> float:
    """
    무엇을: 강제선택 답변(Q4_MORE_SIMILAR/Q4_SAME/Q4_MORE_DIFFERENT)에 따라
    diversity_beta를 ±0.15 조정하고 0.1~0.9로 클립한다.
    (apply_delta 내부에서 이미 처리되지만, 단독 호출이 필요한 경우를 위해 노출한다.)
    """
    beta = float(profile.get("diversity_beta", 0.5))
    if forced_choice_answer == Q4_MORE_SIMILAR:
        return _clamp(beta - BETA_STEP, BETA_MIN, BETA_MAX)
    if forced_choice_answer == Q4_MORE_DIFFERENT:
        return _clamp(beta + BETA_STEP, BETA_MIN, BETA_MAX)
    return beta


def compute_discrepancy(self_report_vector: Dict[str, Any], behavior_corrected_vector: Dict[str, Any]) -> Dict[str, float]:
    """
    무엇을: "현재 값"(clamp(self_report+behavior, 1, 5))과 self_report의 축별
    차이(절댓값).

    왜 behavior 오프셋을 그대로 쓰지 않고 clamp를 다시 거치는가: self_report가
    이미 5(최댓값)인데 behavior가 +1이면, 실제 "현재 값"은 5에서 더 못 올라가므로
    괴리는 0이어야 한다. behavior 오프셋 값을 그대로 괴리로 쓰면 척도 경계에서
    존재하지 않는 괴리를 만들어낸다.
    """
    eff = _effective(self_report_vector, behavior_corrected_vector)
    return {t: abs(eff[t] - float(self_report_vector.get(t, 3.0))) for t in BIG_FIVE_TRAITS}


def check_discrepancy_threshold(discrepancy: Dict[str, float], threshold: float = DIVERGENCE_THRESHOLD) -> bool:
    """축별 괴리 중 하나라도 threshold 이상이면 True ("프로필 이동 이벤트" 트리거)."""
    return any(diff >= threshold for diff in discrepancy.values())


def handle_review_timeout(profile: Dict[str, Any], hours_since_meal: float, failure_count: int) -> Dict[str, Any]:
    """24시간 초과 미제출 또는 2회 이상 실패 시 델타 0으로 안전하게 스킵."""
    if hours_since_meal > REVIEW_TIMEOUT_HOURS:
        logger.warning("[handle_review_timeout] 리뷰 제출 기한(%.0f시간) 초과(%.1f시간) - 델타 0 처리",
                       REVIEW_TIMEOUT_HOURS, hours_since_meal)
    if failure_count >= MAX_REVIEW_FAILURES:
        logger.warning("[handle_review_timeout] 리뷰 제출 %d회 실패 - 델타 0 처리", failure_count)
    return dict(profile)


def run_feedback_pipeline(raw_answers: Dict[str, Any], profile: Dict[str, Any],
                          hours_since_meal: float, failure_count: int = 0) -> Dict[str, Any]:
    """
    무엇을: 리뷰 파싱 -> 델타 추출(LLM) -> EMA 반영(behavior_corrected_vector) ->
    관심사 가중치 갱신(interest_weights) -> 다양성 β 갱신 -> 괴리 계산 -> 임계
    판정까지 전체 파이프라인을 순서대로 실행한다.

    Returns:
        {"profile": 갱신된 profile dict,
         "profile_shift_event": bool,
         "deltas": {trait: 실제_반영량, ...} 또는 스킵된 경우 None}
    """
    should_skip = hours_since_meal > REVIEW_TIMEOUT_HOURS or failure_count >= MAX_REVIEW_FAILURES
    if should_skip:
        return {"profile": handle_review_timeout(profile, hours_since_meal, failure_count),
                "profile_shift_event": False, "deltas": None}

    review_answers = parse_review(raw_answers)
    self_report_vector = profile["self_report_vector"]
    behavior_vector = profile.get("behavior_corrected_vector") or self_report_vector

    try:
        delta = extract_delta(review_answers, self_report_vector, behavior_vector)
    except Exception as exc:
        logger.error("[run_feedback_pipeline] 델타 추출 실패, 델타 0으로 처리: %s", exc)
        delta = {t: 0.0 for t in BIG_FIVE_TRAITS} | {"worked_topics": [], "dead_topics": [], "confidence": 0.0}

    updated_profile = apply_delta(profile, delta, review_answers.get("forced_choice"))
    applied_deltas = updated_profile.pop("_last_deltas", {})

    discrepancy = compute_discrepancy(updated_profile["self_report_vector"], updated_profile["behavior_corrected_vector"])
    profile_shift_event = check_discrepancy_threshold(discrepancy)

    return {"profile": updated_profile, "profile_shift_event": profile_shift_event, "deltas": applied_deltas}
