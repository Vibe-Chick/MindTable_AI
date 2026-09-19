"""
ai_engine/feedback.py
=======================
리뷰 기반 프로필 보정 파이프라인.

식사 후 남긴 리뷰(자유서술 3개 + 강제선택 1개)를 분석해, 온보딩 때 확정된
self_report_vector는 건드리지 않고 별도의 behavior_corrected_vector와 diversity_beta만
점진적으로 보정한다.

파이프라인 개요:
    1) parse_review              : 리뷰 원본 답변 -> 자유서술/강제선택 분리
    2) extract_delta              : 자유서술 + 자기보고 벡터 -> 축별 델타(-1~1) + 근거 인용 (LLM)
    3) clip_delta                 : 델타에 학습률을 곱하고 누적 상한으로 clip
    4) update_behavior_vector     : behavior_corrected_vector에만 델타 누적 반영
    5) update_diversity_beta      : 강제선택 답변으로 다양성 선호도 β 조정
    6) compute_discrepancy        : self_report_vector와 behavior_corrected_vector의 축별 차이
    7) check_discrepancy_threshold: 괴리가 임계값을 넘으면 "프로필 이동 이벤트" 트리거
    8) handle_review_timeout      : 미제출/반복 실패 시 델타 0으로 안전하게 스킵
    9) run_feedback_pipeline      : 위 전체를 순서대로 실행하는 오케스트레이션 함수

개인정보 최소 보관 원칙:
    리뷰 자유서술 원문은 extract_delta 호출 한 번에만 쓰이고, update_behavior_vector 처리가
    끝나면 폐기된다. 프로필에 영구히 남는 것은 원문에서 파생된 숫자(delta)뿐이며, 사용자가
    실제로 작성한 문장 자체는 이 파이프라인을 통과한 뒤 어디에도 저장하지 않는다.

profile dict 형태:
    이 모듈의 함수들은 UserProfile(ai_engine.schemas)을 dataclasses.asdict()한 형태의
    dict를 주고받는다:
        {
          "self_report_vector": {"openness": 4.0, ...},
          "behavior_corrected_vector": {"openness": 4.0, ...},
          "diversity_beta": 0.5,
          ... (name/university/major/interest_tags 등 부가 필드는 그대로 보존)
        }
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .json_utils import _extract_json_object
from .llm_client import _call_llm, _get_chat_backend
from .schemas import BIG_FIVE_TRAITS

logger = logging.getLogger("ai_engine.feedback")

MAX_DELTA_ATTEMPTS = 3  # 최초 시도 1회 + 재시도 2회 (스펙: "최대 2회까지 재요청")
REVIEW_TIMEOUT_HOURS = 24.0
MAX_REVIEW_FAILURES = 2

DELTA_SYSTEM_PROMPT = """당신은 성격심리학 전문가입니다. 사용자가 방금 끝난 랜덤 식사 모임에 대해
남긴 후기(자유서술 3개)를 보고, 이번 경험이 그 사람의 실제 행동 성향을 자기보고 점수 대비
어느 방향으로, 얼마나 움직였는지를 추정합니다.

이미 가지고 있는 자기보고 성격 점수(1~5, Big Five)를 참고하되, 그 값을 직접 바꾸지 말고
"이번 경험이 시사하는 조정 방향과 크기"만 -1.0~+1.0 사이 델타로 추정하세요. 예를 들어
자기보고 개방성이 2였는데 후기에서 "낯선 사람과 새로운 걸 시도해서 좋았다"는 내용이
강하게 드러나면 openness delta는 양수(+)여야 합니다.

각 축마다 그 델타를 뒷받침하는 후기 원문에서 그대로 발췌한 인용구(quote)도 함께 제시하세요.
근거가 될 만한 문장이 없으면 quote는 빈 문자열("")로 하고 delta는 0에 가깝게 잡으세요.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요. 응답의 첫
글자는 반드시 여는 중괄호로 시작하고 마지막 글자는 반드시 닫는 중괄호로 끝나야 합니다.

{
  "openness": {"delta": -1.0~1.0 사이 숫자, "quote": "근거 문장 또는 빈 문자열"},
  "conscientiousness": {"delta": ..., "quote": "..."},
  "extraversion": {"delta": ..., "quote": "..."},
  "agreeableness": {"delta": ..., "quote": "..."},
  "neuroticism": {"delta": ..., "quote": "..."}
}

### 예시
자기보고 성격 점수: {"openness": 2, "conscientiousness": 4, "extraversion": 2, "agreeableness": 4, "neuroticism": 3}
후기:
- r1: "낯선 전공 사람들이랑 얘기하는 게 생각보다 재밌었고, 다음엔 더 새로운 모임에도 나가보고 싶어요."
- r2: "그래도 낯가림 때문에 처음엔 많이 긴장했어요."
- r3: "다들 친절해서 편하게 얘기할 수 있었어요."

출력:
{"openness": {"delta": 0.4, "quote": "다음엔 더 새로운 모임에도 나가보고 싶어요"}, "conscientiousness": {"delta": 0.0, "quote": ""}, "extraversion": {"delta": 0.2, "quote": "다들 친절해서 편하게 얘기할 수 있었어요"}, "agreeableness": {"delta": 0.1, "quote": "다들 친절해서"}, "neuroticism": {"delta": -0.1, "quote": "처음엔 많이 긴장했어요"}}
"""


def parse_review(raw_answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    무엇을: 식사 후 리뷰 설문 원본 답변(자유서술 3개 r1~r3 + 강제선택 1개 r4_choice)을
    자유서술 텍스트 리스트와 강제선택 답변으로 분리해 반환한다.

    왜: extract_delta는 자유서술 텍스트만 LLM에 넘겨 성격 델타를 추론하고, 강제선택 답변은
    별도로 update_diversity_beta가 규칙 기반으로 처리한다. 두 흐름이 쓰는 재료가 다르므로
    입력 단계에서 미리 분리해두면 이후 함수들의 인터페이스가 각자 필요한 데이터만 받도록
    깔끔해진다.
    """
    free_text = [
        str(raw_answers.get("r1", "")),
        str(raw_answers.get("r2", "")),
        str(raw_answers.get("r3", "")),
    ]
    forced_choice = str(raw_answers.get("r4_choice", ""))
    return {"free_text": free_text, "forced_choice": forced_choice}


def _build_delta_user_prompt(review_answers: Dict[str, Any], self_report_vector: Dict[str, Any]) -> str:
    free_text: List[str] = review_answers.get("free_text", [])
    lines = [f"- r{i + 1}: \"{text}\"" for i, text in enumerate(free_text)]
    self_report_json = json.dumps(self_report_vector, ensure_ascii=False)
    return (
        f"자기보고 성격 점수: {self_report_json}\n"
        "후기:\n" + "\n".join(lines) + "\n\n위 형식의 JSON으로만 응답해줘."
    )


def _validate_delta_response(data: Dict[str, Any]) -> None:
    """
    무엇을: extract_delta의 LLM 응답이 5축 모두 {delta, quote}를 갖추고, delta가 -1~1
    범위인지 검증한다.
    왜: extract_profile의 _validate_profile과 같은 이유 - LLM 출력은 확률적이므로 스키마
    이탈이 드물게 생기고, 여기서 걸러야 update_behavior_vector가 잘못된 값으로 프로필을
    오염시키는 걸 막을 수 있다.
    """
    for trait in BIG_FIVE_TRAITS:
        if trait not in data:
            raise ValueError(f"필수 필드 누락: {trait}")
        entry = data[trait]
        if not isinstance(entry, dict) or "delta" not in entry or "quote" not in entry:
            raise ValueError(f"'{trait}' 항목 형식이 올바르지 않음: {entry!r}")
        delta = entry["delta"]
        if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not (-1.0 <= delta <= 1.0):
            raise ValueError(f"'{trait}'.delta가 -1~1 범위를 벗어남: {delta!r}")


def extract_delta(review_answers: Dict[str, Any], self_report_vector: Dict[str, Any]) -> Dict[str, Any]:
    """
    무엇을: 리뷰 자유서술 답변과 기존 자기보고 벡터를 LLM에 함께 전달해, Big Five 5축
    각각에 대해 -1~+1 범위의 델타(보정값)와 그 근거가 된 인용구를 구조화된 JSON으로
    추출한다. 반환 형태: {"openness": {"delta": 0.3, "quote": "..."}, ...} (5축 전부).

    왜 자기보고 벡터를 함께 넘기는가: 같은 문장이라도 이미 개방성이 높은 사람에게는
    "평소와 비슷한 경험"일 수 있고, 개방성이 낮은 사람에게는 "평소와 다른 도전"일 수
    있다. 기존 점수를 기준점으로 줘야 LLM이 "이번 경험이 그 사람 기준으로 어느 방향으로
    움직였는지"를 상대적으로 판단할 수 있다.

    왜 인용구(quote)를 함께 요구하는가: delta 숫자만 받으면 왜 그렇게 나왔는지 검증할
    방법이 없다. 원문 발췌를 강제하면 LLM이 실제로 텍스트에 근거해 판단했는지 확인할 수
    있고(환각 방지에 도움), 필요하면 사용자에게 "이 문장 때문에 이렇게 조정됐다"고
    설명할 수도 있다.

    왜 재시도 로직이 필요한지: extract_profile과 동일한 이유로, LLM 출력이 스키마를
    벗어나는 경우를 대비해 최대 2회까지 재요청하고 그래도 실패하면 예외를 던진다.

    Raises:
        RuntimeError: 최대 재시도 횟수를 넘겨도 유효한 델타를 얻지 못한 경우.
    """
    if _get_chat_backend() is None:
        logger.warning("[extract_delta] 사용 가능한 LLM 백엔드가 없어 델타를 전부 0으로 처리합니다.")
        return {trait: {"delta": 0.0, "quote": ""} for trait in BIG_FIVE_TRAITS}

    user_prompt = _build_delta_user_prompt(review_answers, self_report_vector)
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_DELTA_ATTEMPTS + 1):
        try:
            raw_text = _call_llm(user_prompt, max_tokens=700, system_prompt=DELTA_SYSTEM_PROMPT)
            json_text = _extract_json_object(raw_text)
            data = json.loads(json_text)
            _validate_delta_response(data)
            return {
                trait: {"delta": float(data[trait]["delta"]), "quote": str(data[trait]["quote"])}
                for trait in BIG_FIVE_TRAITS
            }
        except Exception as exc:
            last_error = exc
            logger.warning(f"[extract_delta] 시도 {attempt}/{MAX_DELTA_ATTEMPTS} 실패: {exc}")

    raise RuntimeError(f"extract_delta 실패: 최대 재시도({MAX_DELTA_ATTEMPTS}회) 초과. 마지막 오류: {last_error}")


def clip_delta(
    raw_delta: float,
    learning_rate: float = 0.5,
    accumulated_cap: float = 1.5,
    current_accumulated: float = 0.0,
) -> float:
    """
    무엇을: LLM이 추정한 원본 델타(raw_delta, -1~1)에 학습률을 곱해 완만하게 줄이고,
    지금까지 누적된 보정값(current_accumulated, 이 축에서 self_report_vector로부터 이미
    벌어진 절댓값 거리)과 합쳤을 때 상한(accumulated_cap)을 넘지 않도록 다시 자른다.

    왜 학습률(learning_rate)이 필요한가: 리뷰 한 번은 표본 크기 1의 관찰치일 뿐이다.
    한 번의 식사 경험(우연히 그날 컨디션이 좋았거나 나빴던 것일 수도 있음)만으로 성격
    프로필 전체가 크게 흔들리면, 다음 매칭의 성격 유사도 계산이 노이즈에 휘둘리게 된다.
    학습률(기본 0.5)을 곱해 "이번 리뷰가 시사하는 방향으로 절반만" 반영함으로써, 여러
    번의 리뷰가 일관되게 쌓여야 프로필이 실제로 크게 이동하는 점진적(온라인) 학습 구조를
    만든다.

    왜 누적 상한(accumulated_cap)이 필요한가: 학습률로 매 회 반영폭을 줄여도, 계속 같은
    방향의 리뷰가 쌓이면 behavior_corrected_vector가 self_report_vector에서 한없이
    멀어질 수 있다. Big Five 척도 자체가 1~5(폭 4)이므로, 자기보고 대비 ±1.5 이상
    벌어지는 것은 "완전히 다른 사람"이라고 볼 정도로 과도한 괴리다. 이 상한을 넘기지
    않도록 클리핑해 극단적 드리프트를 방지한다.
    """
    scaled = raw_delta * learning_rate
    remaining_room = max(0.0, accumulated_cap - current_accumulated)
    if scaled > 0:
        return float(min(scaled, remaining_room))
    return float(max(scaled, -remaining_room))


def update_behavior_vector(profile: Dict[str, Any], clipped_deltas: Dict[str, float]) -> Dict[str, Any]:
    """
    무엇을: self_report_vector는 절대 건드리지 않고, behavior_corrected_vector에만
    clipped_deltas를 더해 누적 반영한 새 profile dict를 반환한다.

    왜 self_report_vector를 불변으로 두는가: self_report_vector는 온보딩 시점에
    extract_profile()이 만든 "이 사람이 스스로를 어떻게 인식하는가"의 기준점이다(profile.py
    상단 docstring 참고). 매칭 시스템이 이 기준점 자체를 계속 덮어쓰면, 두 벡터를 비교해
    "자기 인식과 실제 행동의 괴리"를 측정한다는 compute_discrepancy의 개념 자체가 성립하지
    않는다. 따라서 behavior_corrected_vector라는 별도의 가변 벡터에만 보정치를 누적한다.

    개인정보 최소 보관 원칙: 이 함수가 받는 clipped_deltas는 이미 리뷰 원문에서 파생된
    숫자 값일 뿐이다. 리뷰 원문(자유서술 텍스트) 자체는 extract_delta 호출 이후로는
    어디에도 저장하지 않고 이 함수 처리가 끝나면 완전히 폐기된다 - 프로필에 남는 것은
    "얼마나 이동했는가"라는 파생값뿐, 사용자가 실제로 쓴 문장은 보관하지 않는다.
    """
    updated = dict(profile)
    self_report_vector = dict(profile["self_report_vector"])  # 참조만 복사, 값은 그대로 유지(불변)
    behavior_vector = dict(profile.get("behavior_corrected_vector") or self_report_vector)

    for trait in BIG_FIVE_TRAITS:
        current = behavior_vector.get(trait, self_report_vector.get(trait, 0.0))
        behavior_vector[trait] = current + clipped_deltas.get(trait, 0.0)

    updated["self_report_vector"] = self_report_vector
    updated["behavior_corrected_vector"] = behavior_vector
    return updated


def update_diversity_beta(profile: Dict[str, Any], forced_choice_answer: str) -> float:
    """
    무엇을: 리뷰의 강제선택 문항("이번에 낯선 배경의 사람과 만난 게 어땠나요?" 류) 답변에
    따라 다양성 선호도 β를 ±0.15 조정하고 0.1~0.9 범위로 클립해 반환한다.

    왜 β를 별도로 두는가: pair_score의 diversity_score 가중치(w_diversity)는 서비스
    전체에 고정된 값이지만, 실제로는 사람마다 "낯선 배경의 사람"을 만났을 때의 만족도가
    다르다. 리뷰에서 다양성 경험에 대한 명시적 선호를 확인할 때마다 그 사람만의 β를
    조금씩 조정해두면, 향후 개인화된 가중치(예: pair_score의 w_diversity를 사용자별 β로
    대체)로 확장하기 쉬워진다.

    왜 0.1~0.9로 클립하는가: β가 0이나 1처럼 극단으로 가면 "다양성을 아예 무시" 또는
    "다양성만 본다"는 뜻이 되어 매칭이 성격/관심사 유사도를 완전히 무시하게 될 수 있다.
    항상 최소한의 여지를 남겨두기 위해 양 끝을 잘라낸다.
    """
    current_beta = float(profile.get("diversity_beta", 0.5))

    positive_markers = ["좋았", "재밌", "재미있", "새로웠", "신선", "만족", "또 만나고", "괜찮았"]
    negative_markers = ["불편", "어색", "별로", "힘들", "피곤", "안 맞", "부담"]

    answer = forced_choice_answer or ""
    if any(marker in answer for marker in positive_markers):
        current_beta += 0.15
    elif any(marker in answer for marker in negative_markers):
        current_beta -= 0.15

    return float(max(0.1, min(0.9, current_beta)))


def compute_discrepancy(self_report_vector: Dict[str, Any], behavior_corrected_vector: Dict[str, Any]) -> Dict[str, float]:
    """
    무엇을: self_report_vector와 behavior_corrected_vector의 축별 차이(절댓값)를 계산해
    반환한다.
    왜: 이 차이가 "이 사람이 스스로 생각하는 성격"과 "실제 식사 모임에서 드러난 성향"이
    얼마나 벌어졌는지를 나타내는 지표가 된다. 하나로 합산하지 않고 축별로 남겨두면 어떤
    축에서 괴리가 발생했는지 원인을 파악하기 쉽다.
    """
    return {
        trait: abs(float(behavior_corrected_vector.get(trait, 0.0)) - float(self_report_vector.get(trait, 0.0)))
        for trait in BIG_FIVE_TRAITS
    }


def check_discrepancy_threshold(discrepancy: Dict[str, float], threshold: float = 1.0) -> bool:
    """
    무엇을: 축별 괴리 중 하나라도 threshold(기본 1.0) 이상이면 True를 반환한다.
    왜: 임계값을 넘는다는 것은 "이 사람의 실제 행동이 자기보고와 상당히 다르게 나타나고
    있다"는 신호로, 매칭 알고리즘이 self_report_vector 대신 behavior_corrected_vector를
    우선 사용하도록 전환하거나, 운영진에게 "프로필 이동 이벤트"를 알려 재검토를 유도하는
    트리거로 쓰기 위함이다.
    """
    return any(diff >= threshold for diff in discrepancy.values())


def handle_review_timeout(profile: Dict[str, Any], hours_since_meal: float, failure_count: int) -> Dict[str, Any]:
    """
    무엇을: (1) 식사 후 24시간이 지나도록 리뷰 미제출, 또는 (2) 리뷰 제출 시도가 2회 이상
    파싱/검증에 실패한 경우, 이번 회차의 델타를 0으로 처리(=behavior_corrected_vector를
    바꾸지 않음)하고 profile을 그대로(복사본으로) 반환한다.

    왜 그냥 실패시키지 않고 "델타 0"으로 처리하는가: 리뷰 미제출/반복 실패는 사용자
    귀책일 수도, 시스템 문제일 수도 있다. 어느 쪽이든 이 한 번의 실패 때문에 프로필이
    깨지거나 사용자가 다음 매칭에서 배제되면 안 되므로, "이번 회차는 그냥 스킵하고
    다음 매칭부터 정상 참여"로 안전하게 되돌리는 것이 서비스 신뢰도에 유리하다.
    """
    if hours_since_meal > REVIEW_TIMEOUT_HOURS:
        logger.warning(
            f"[handle_review_timeout] 리뷰 제출 기한({REVIEW_TIMEOUT_HOURS:.0f}시간) 초과"
            f"({hours_since_meal:.1f}시간) - 이번 회차 델타 0 처리"
        )
    if failure_count >= MAX_REVIEW_FAILURES:
        logger.warning(f"[handle_review_timeout] 리뷰 제출 {failure_count}회 실패 - 이번 회차 델타 0 처리")
    return dict(profile)


def run_feedback_pipeline(
    raw_answers: Dict[str, Any],
    profile: Dict[str, Any],
    hours_since_meal: float,
    failure_count: int = 0,
) -> Dict[str, Any]:
    """
    무엇을: 리뷰 파싱 -> 델타 추출 -> 클리핑 -> 행동 벡터 갱신 -> 다양성 β 갱신 -> 괴리도 계산 ->
    임계값 판정까지, 리뷰 기반 프로필 보정 파이프라인 전체를 순서대로 실행한다.

    왜 오케스트레이션 함수를 분리했는가: 백엔드 API(예: POST /review/submit) 핸들러가
    이 함수 하나만 호출하면 되도록 해서, 각 단계(파싱/LLM 호출/클리핑/누적)의 세부 구현이
    API 계층으로 새어나가지 않게 한다.

    Returns:
        {
          "profile": 갱신된 profile dict,
          "profile_shift_event": bool,  # 임계 초과로 "프로필 이동 이벤트"가 트리거됐는지
          "deltas": {trait: clipped_delta, ...} 또는 타임아웃/실패로 스킵된 경우 None,
        }
    """
    should_skip = hours_since_meal > REVIEW_TIMEOUT_HOURS or failure_count >= MAX_REVIEW_FAILURES
    if should_skip:
        updated_profile = handle_review_timeout(profile, hours_since_meal, failure_count)
        return {"profile": updated_profile, "profile_shift_event": False, "deltas": None}

    review_answers = parse_review(raw_answers)
    self_report_vector = profile["self_report_vector"]
    behavior_vector = profile.get("behavior_corrected_vector") or self_report_vector

    try:
        raw_deltas = extract_delta(review_answers, self_report_vector)
    except Exception as exc:
        logger.error(f"[run_feedback_pipeline] 델타 추출 실패, 이번 회차 델타 0으로 처리: {exc}")
        raw_deltas = {trait: {"delta": 0.0, "quote": ""} for trait in BIG_FIVE_TRAITS}

    # 각 축에서 이미 self_report_vector로부터 얼마나 벌어져 있는지(현재 누적량)를 구해
    # clip_delta의 accumulated_cap 계산에 넘긴다.
    current_accumulated = {
        trait: abs(float(behavior_vector.get(trait, self_report_vector.get(trait, 0.0))) - float(self_report_vector.get(trait, 0.0)))
        for trait in BIG_FIVE_TRAITS
    }

    clipped_deltas = {
        trait: clip_delta(raw_deltas[trait]["delta"], current_accumulated=current_accumulated[trait])
        for trait in BIG_FIVE_TRAITS
    }

    updated_profile = update_behavior_vector(profile, clipped_deltas)
    updated_profile["diversity_beta"] = update_diversity_beta(updated_profile, review_answers.get("forced_choice", ""))

    discrepancy = compute_discrepancy(updated_profile["self_report_vector"], updated_profile["behavior_corrected_vector"])
    profile_shift_event = check_discrepancy_threshold(discrepancy)

    return {"profile": updated_profile, "profile_shift_event": profile_shift_event, "deltas": clipped_deltas}
