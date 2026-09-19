"""
ai_engine/profile.py
======================
온보딩 답변 -> 자기보고 성격 벡터(self_report_vector) 추출.

통합 결정(INTEGRATION_REPORT.md)에 따라 김주환 브랜치의 `AI/src/extract.py` +
`AI/src/prompts.py`의 핵심 기법 - **인용문을 답변 원문과 문자열 대조하는 근거
검증(grounding)** - 을 그대로 이식했다. 기존 jaebin의 온보딩 설문 형식(q1~q4_choice,
자기소개/스트레스 해소법/좋아하는 활동/역할 선택)과 API 경계 필드명(영문 풀네임)은
그대로 유지한다.

*** 중요 ***
이 함수가 만들어내는 Big Five 5축 점수는 UserProfile.self_report_vector로 감싸진 뒤에는
온보딩 시점 이후로 **절대 변하지 않는 고정 기준점**이다. 식사 후 리뷰 피드백
(ai_engine/feedback.py)은 이 값을 직접 수정하지 않고, 별도의 behavior_corrected_vector와
interest_weights만 움직인다.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Dict, Optional

from . import llm_client
from .schemas import BIG_FIVE_TRAITS, TAGS

logger = logging.getLogger("ai_engine.profile")

MAX_EXTRACT_ATTEMPTS = 2  # llm_client.call() 자체도 내부적으로 재시도하므로, 여기는 가볍게

_SCORE = {"type": "integer", "minimum": 1, "maximum": 5}
_QUOTE = {"type": "string"}

# 평탄한 스키마 - 축마다 점수와 '원문 인용'을 쌍으로 받는다. 중첩 object는 haiku급
# 모델에서 절반쯤 누락되는 경우가 실측됐다(김주환 브랜치 schema.py 근거).
EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "openness": _SCORE, "openness_quote": _QUOTE,
        "conscientiousness": _SCORE, "conscientiousness_quote": _QUOTE,
        "extraversion": _SCORE, "extraversion_quote": _QUOTE,
        "agreeableness": _SCORE, "agreeableness_quote": _QUOTE,
        "neuroticism": _SCORE, "neuroticism_quote": _QUOTE,
        "interest_tags": {"type": "array", "items": {"type": "string"},
                          "minItems": 1, "maxItems": 4},
        "interest_quotes": {"type": "array", "items": {"type": "string"},
                            "minItems": 1, "maxItems": 4},
        "tags": {"type": "array",
                 "items": {"type": "string", "enum": list(TAGS)},
                 "minItems": 1, "maxItems": 3},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["openness", "openness_quote",
                 "conscientiousness", "conscientiousness_quote",
                 "extraversion", "extraversion_quote",
                 "agreeableness", "agreeableness_quote",
                 "neuroticism", "neuroticism_quote",
                 "interest_tags", "interest_quotes", "tags", "confidence"],
}


EXTRACT_SYSTEM_PROMPT = """\
너는 심리측정 보조 도구다. 대학생의 자유서술 답변을 읽고 Big Five 5개 축을
1~5점으로 채점하고, 관심사 키워드를 뽑는다.

**모든 판단에는 원문 인용이 따라야 한다.**
<축>_quote 에는 답변에서 **글자 그대로 복사한** 구절을 넣는다(20~40자).
요약·해석·바꿔쓰기 금지. 답변에 없는 문장을 쓰면 그 점수는 버려진다.

채점 기준:
- 3점이 기본값이다.
- 그 축을 직접 뒷받침하는 인용이 없으면 quote를 ""로 두고 점수는 3으로 한다.
- 다른 축의 근거를 끌어다 쓰지 마라.
  "새 기술에 빠졌다"는 개방성의 근거이지 성실성의 근거가 아니다.
- 1점/5점은 명확한 인용이 있을 때만.
- 신경성(N)은 답변에서 직접 드러나는 경우가 드물다. 대개 ""에 3점이다.

관심사(interest_tags):
- interest_tags와 interest_quotes는 **길이가 같아야 한다.** i번째 키워드의
  근거가 i번째 인용이다.
- 답변에 실제로 등장한 것만. 일반명사로 정규화한다
  ("주말마다 북한산 간다" -> "등산", 인용은 "주말마다 북한산 간다").
- **개수를 채우려고 지어내지 마라.** 근거가 하나뿐이면 하나만 낸다.

tags:
- 위 관심사를 정해진 태그 목록에서 1~3개로 분류한다. 목록에 없는 값 금지.
- 자유 키워드가 하나뿐이어도 그 성격에 맞는 태그를 고른다.
  예: "홈서버/리눅스" -> 기술 / "클라이밍" -> 운동 / "학회 네트워킹" -> 학술
- 억지로 3개를 채우지 마라. 근거 있는 것만.

confidence는 답변이 짧거나 모호하면 낮춘다.

반드시 위 스키마의 JSON으로만 응답하라.
"""


def _build_extract_user_prompt(answers: Dict[str, str]) -> str:
    """온보딩 답변 dict(q1~q4_choice)를 프롬프트용 텍스트로 변환한다."""
    return (
        "[Q1/자기소개] 낯선 사람들과 있을 때 vs 혼자 있을 때, 언제 에너지가 더 차오르나요?\n"
        f"{answers.get('q1', '').strip() or '(무응답)'}\n\n"
        "[Q2/스트레스 해소법] 스트레스를 받을 때 보통 어떻게 푸나요?\n"
        f"{answers.get('q2', '').strip() or '(무응답)'}\n\n"
        "[Q3/좋아하는 활동] 요즘 관심 생긴 주제나 즐겨 하는 활동이 있나요?\n"
        f"{answers.get('q3', '').strip() or '(무응답)'}\n\n"
        f"[Q4/강제선택] 그룹에서 편한 역할: {answers.get('q4_choice', '').strip() or '(무응답)'}\n"
    )


def _norm(text: str) -> str:
    """공백·줄바꿈을 지워서 인용 대조용으로 정규화."""
    return "".join((text or "").split())


def _source_text(answers: Dict[str, str]) -> str:
    return " ".join(str(answers.get(k, "") or "") for k in ("q1", "q2", "q3", "q4_choice"))


def _ground_and_clean(out: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
    """
    무엇을: LLM 응답의 각 점수/관심사를 답변 원문과 대조해, 근거 없는 판단을 걸러낸다
    (김주환 브랜치 extract.py의 핵심 방어 로직 이식).

    왜: 모델은 "요약"을 "인용"이라고 내놓는 경향이 실측으로 확인됐다. 그건 근거가
    아니다. 인용문이 비었거나 답변 원문에 실제로 없으면, 그 축 점수는 신뢰할 수
    없으므로 중립값(3점)으로 되돌리고 인용은 비운다. 관심사도 동일하게, 근거 문장이
    원문에 없으면 그 키워드 자체를 버린다(모델이 지어낸 관심사를 걸러내는 유일한
    방법은 "원문에 있었는가"뿐이다).
    """
    src = _norm(_source_text(answers))
    scores: Dict[str, int] = {}
    dropped = []

    for trait in BIG_FIVE_TRAITS:
        quote = (out.get(trait + "_quote") or "").strip()
        score = int(out[trait])
        grounded = bool(quote) and _norm(quote) in src
        if not grounded:
            if score != 3:
                dropped.append(f"{trait}:{score}->3" + ("" if not quote else "(인용불일치)"))
            score = 3
        scores[trait] = score

    kws = out.get("interest_tags") or []
    kqs = out.get("interest_quotes") or []
    interests = []
    for i, kw in enumerate(kws):
        kw = (kw or "").strip()
        q = (kqs[i] if i < len(kqs) else "").strip()
        if not kw:
            continue
        if not q or _norm(q) not in src:
            dropped.append(f"관심사 '{kw}' 근거없음->제거")
            continue
        interests.append(kw)

    tags = [t for t in (out.get("tags") or []) if t in TAGS]

    result: Dict[str, Any] = dict(scores)
    result["interest_tags"] = interests
    result["tags"] = tags
    result["confidence"] = float(out.get("confidence", 0.5) or 0.5)
    if dropped:
        result["dropped"] = dropped
        logger.info("[extract_profile] 근거 미달로 제거: %s", "; ".join(dropped))
    return result


def _fallback_profile(answers: Dict[str, str]) -> Dict[str, Any]:
    """
    무엇을: LLM 백엔드가 설정되어 있지 않을 때 쓰는 규칙 기반 대체 프로필.
    왜: 데모/개발 환경에서 API 키 없이도 파이프라인이 끝까지 동작해야 한다. 답변
    텍스트를 해시로 시드 고정해 같은 입력에는 항상 같은 결과가 나오게 한다.
    """
    combined = " ".join(str(v) for v in answers.values())
    seed = int(hashlib.sha256(combined.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    profile: Dict[str, Any] = {trait: rng.randint(1, 5) for trait in BIG_FIVE_TRAITS}

    words = [w.strip(".,!?\"'") for w in combined.split() if len(w.strip(".,!?\"'")) > 1]
    interest_tags = list(dict.fromkeys(words))[:3] or ["새로운활동"]
    profile["interest_tags"] = interest_tags
    profile["tags"] = [TAGS[seed % len(TAGS)]]
    profile["confidence"] = 0.3
    return profile


def extract_profile(answers: Dict[str, str]) -> Dict[str, Any]:
    """
    무엇을: 온보딩 설문 답변 4개를 LLM에 전달해 Big Five 5축 점수(1~5) + 관심사
    키워드 + 고정 태그(tags) + confidence를 추출한다. 각 판단은 답변 원문과
    문자열 대조로 근거를 검증하고, 근거 없는 판단은 중립값으로 되돌리거나 버린다.

    왜 이렇게 설계했는지 (심리학적 근거): Big Five(OCEAN) 모델은 성격심리학에서
    가장 널리 검증된 특성 이론이며, 자유 서술형 텍스트에서도 어휘/화제 선택을 통해
    비교적 안정적으로 추론 가능하다.

    왜 인용 그라운딩이 필요한지 (통합 결정의 핵심): 프롬프트로 "지어내지 마라"라고
    지시하는 것만으로는 모델이 요약을 인용이라 내놓는 걸 막지 못한다(실측 확인).
    인용문이 답변 원문에 실제로 있는지 코드로 대조해야, 근거 없는 고점수/저점수나
    지어낸 관심사가 프로필에 섞이지 않는다.

    Raises:
        RuntimeError: LLM 백엔드가 설정되어 있는데도 호출이 끝내 실패한 경우.
        (백엔드 자체가 설정되어 있지 않으면 규칙 기반 폴백으로 조용히 대체한다.)
    """
    if not llm_client.is_configured():
        return _fallback_profile(answers)

    user_prompt = _build_extract_user_prompt(answers)
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_EXTRACT_ATTEMPTS + 1):
        try:
            out = llm_client.call(user_prompt, EXTRACTION_SCHEMA,
                                  system=EXTRACT_SYSTEM_PROMPT, temp=0.0,
                                  tag="extract:" + hashlib.sha256(user_prompt.encode()).hexdigest()[:8])
            return _ground_and_clean(out, answers)
        except Exception as exc:
            last_error = exc
            logger.warning("[extract_profile] 시도 %d/%d 실패: %s", attempt, MAX_EXTRACT_ATTEMPTS, exc)

    raise RuntimeError(f"extract_profile 실패: 최대 재시도({MAX_EXTRACT_ATTEMPTS}회) 초과. 마지막 오류: {last_error}")
