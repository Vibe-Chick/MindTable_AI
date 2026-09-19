"""
ai_engine/profile.py
======================
온보딩 답변 -> 자기보고 성격 벡터(self_report_vector) 추출.

matching_engine.py의 extract_profile 및 관련 헬퍼를 동작 변경 없이 그대로 옮겼다
(실제 학교 게이트웨이로 이미 검증된 로직).

*** 중요 ***
이 함수가 만들어내는 Big Five 5축 점수는 UserProfile.self_report_vector로 감싸진 뒤에는
온보딩 시점 이후로 **절대 변하지 않는 고정 기준점**이다. 식사 후 리뷰 피드백
(ai_engine/feedback.py)은 이 값을 직접 수정하지 않고, 별도의 behavior_corrected_vector만
움직인다. extract_profile을 다시 호출해 self_report_vector를 새로 만드는 것은 "재온보딩"에
해당하는 완전히 다른 작업이며, 일반적인 리뷰 처리 흐름에서는 절대 일어나서는 안 된다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import Any, Dict, Optional

from .json_utils import _extract_json_object
from .llm_client import _call_llm, _get_chat_backend
from .schemas import BIG_FIVE_TRAITS

logger = logging.getLogger("ai_engine.profile")

MAX_EXTRACT_ATTEMPTS = 3  # 최초 시도 1회 + 재시도 2회 (스펙: "최대 2회까지 재요청")


EXTRACT_SYSTEM_PROMPT = """당신은 성격심리학 전문가이자 설문 분석가입니다.
사용자의 온보딩 답변(자기소개/스트레스 해소법/좋아하는 활동 각 1개 + 자기 인식 선택 1개)을 분석해
Big Five(OCEAN) 성격 5요인 점수(1~5 정수)와 관심사 키워드 3개를 추출합니다.

반드시 아래 JSON 형식으로만 응답하세요. 인사말, 설명, 마크다운 코드블록(```) 등
JSON 이외의 어떤 텍스트도 포함하지 마세요. 응답의 첫 글자는 반드시 여는 중괄호로 시작하고
마지막 글자는 반드시 닫는 중괄호로 끝나야 합니다.

{
  "openness": 1~5 정수,
  "conscientiousness": 1~5 정수,
  "extraversion": 1~5 정수,
  "agreeableness": 1~5 정수,
  "neuroticism": 1~5 정수,
  "interest_tags": ["키워드1", "키워드2", "키워드3"]
}

### 예시 1
답변:
- q1(자기소개): "새로운 사람 만나는 걸 좋아하고 주말마다 등산이나 캠핑을 다녀요. 계획 세우는 것도 좋아해요."
- q2(스트레스 해소법): "친구들이랑 수다 떨면서 풀어요"
- q3(좋아하는 활동): "보드게임, 여행 브이로그 촬영"
- q4_choice: "분위기메이커"

출력:
{"openness": 4, "conscientiousness": 4, "extraversion": 5, "agreeableness": 4, "neuroticism": 2, "interest_tags": ["등산", "보드게임", "여행"]}

### 예시 2
답변:
- q1(자기소개): "혼자 책 읽거나 코딩하면서 시간 보내는 걸 좋아합니다. 낯선 사람 앞에서는 조금 긴장하는 편이에요."
- q2(스트레스 해소법): "혼자 산책하며 생각을 정리해요"
- q3(좋아하는 활동): "인공지능 스터디, 독서"
- q4_choice: "리더"

출력:
{"openness": 4, "conscientiousness": 5, "extraversion": 2, "agreeableness": 3, "neuroticism": 3, "interest_tags": ["코딩", "인공지능", "독서"]}
"""


def _build_extract_user_prompt(answers: Dict[str, str]) -> str:
    """온보딩 답변 dict를 프롬프트에 넣을 사람이 읽기 좋은 텍스트 블록으로 변환한다."""
    return (
        "아래 사용자의 답변을 분석해서 지정된 JSON 형식으로만 응답해줘.\n\n"
        f"- q1(자기소개): \"{answers.get('q1', '')}\"\n"
        f"- q2(스트레스 해소법): \"{answers.get('q2', '')}\"\n"
        f"- q3(좋아하는 활동): \"{answers.get('q3', '')}\"\n"
        f"- q4_choice: \"{answers.get('q4_choice', '')}\"\n"
    )


def _validate_profile(profile: Dict[str, Any]) -> None:
    """
    무엇을: extract_profile의 LLM 응답이 필수 스키마(성격 5축 + 관심사 3개)를 만족하는지 검증한다.
    왜: LLM은 확률적으로 출력하므로 필드 누락/범위 이탈/타입 오류가 드물게 발생할 수 있다.
        매칭 로직(personality_similarity, get_embedding 등) 전체가 이 스키마를 전제로 동작하므로,
        여기서 걸러내지 않으면 파이프라인 뒤쪽에서 원인 파악이 어려운 오류로 번진다.
    """
    for trait in BIG_FIVE_TRAITS:
        if trait not in profile:
            raise ValueError(f"필수 필드 누락: {trait}")
        value = profile[trait]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not (1 <= value <= 5):
            raise ValueError(f"'{trait}' 값이 유효하지 않음(1~5 정수여야 함): {value!r}")

    if "interest_tags" not in profile:
        raise ValueError("필수 필드 누락: interest_tags")
    tags = profile["interest_tags"]
    if not isinstance(tags, list) or len(tags) < 3 or not all(isinstance(t, str) and t.strip() for t in tags):
        raise ValueError(f"interest_tags는 비어있지 않은 문자열 3개 이상의 리스트여야 함: {tags!r}")
    profile["interest_tags"] = tags[:3]


def _fallback_profile(answers: Dict[str, str]) -> Dict[str, Any]:
    """
    무엇을: Anthropic API 키가 없거나 호출이 끝내 실패했을 때 사용하는 규칙 기반 대체 프로필 생성.
    왜: 데모/개발 환경에서 API 키 없이도 파이프라인이 죽지 않고 끝까지 동작해야 하기 때문이다.
        답변 텍스트를 해시로 시드 고정하여, 같은 입력에는 항상 같은 점수가 나오도록 결정론적으로 만들었다
        (재현 가능성 확보 - 매칭 결과가 실행마다 달라지면 디버깅이 불가능해진다).
    """
    combined = " ".join(str(v) for v in answers.values())
    seed = int(hashlib.sha256(combined.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    profile: Dict[str, Any] = {trait: rng.randint(1, 5) for trait in BIG_FIVE_TRAITS}

    # 아주 단순한 키워드 후보 추출(실서비스에서는 형태소 분석기/LLM 사용을 권장).
    words = [w.strip(".,!?\"'") for w in combined.split() if len(w.strip(".,!?\"'")) > 1]
    tags = list(dict.fromkeys(words))[:3]
    while len(tags) < 3:
        tags.append("새로운활동")
    profile["interest_tags"] = tags
    return profile


def extract_profile(answers: Dict[str, str]) -> Dict[str, Any]:
    """
    무엇을: 온보딩 설문 답변 4개를 Claude API에 전달하여 Big Five 성격 5요인 점수(1~5)와
    관심사 키워드 3개를 구조화된 JSON으로 추출한다. 이 결과로 만들어지는 벡터는
    UserProfile.self_report_vector로 감싸진 뒤 **온보딩 이후 절대 변하지 않는 자기보고
    벡터**가 된다 (모듈 상단 docstring 참고).

    왜 이렇게 설계했는지 (심리학적 근거):
    - Big Five(OCEAN) 모델은 성격심리학에서 가장 널리 검증된 특성 이론이며, 자유 서술형
      텍스트에서도 어휘/화제 선택을 통해 비교적 안정적으로 추론 가능하다는 선행 연구
      (예: Pennebaker & King의 언어-성격 상관 연구, LIWC 계열 텍스트 분석)가 뒷받침한다.
    - 정량 설문지 대신 "자연어 답변 4개 + LLM 추론"을 쓰는 이유는, 온보딩 문항 수를
      최소화해 이탈률을 낮추면서도(마찰 감소) 성격 신호는 놓치지 않기 위함이다.
    - few-shot 예시를 프롬프트에 포함한 이유는, 자유생성 모델은 스키마가 흔들리기 쉬운데
      입력-출력 쌍을 미리 보여주면 출력 형식(JSON 키 이름, 값 범위)이 훨씬 안정적으로
      고정되기 때문이다(구조화 출력의 few-shot 앵커링 효과).

    왜 재시도 로직이 필요한지 (알고리즘적 근거):
    - LLM 출력은 확률적이므로 JSON 파싱 실패나 필드 누락이 드물게 발생한다. 매칭 파이프라인
      전체가 사용자 한 명의 파싱 실패로 죽어서는 안 되므로, 최대 2회까지 재요청하고
      그래도 실패하면 상위 호출자가 처리할 수 있도록 명시적으로 예외를 던진다.

    Raises:
        RuntimeError: 최대 재시도 횟수를 넘겨도 유효한 프로필을 얻지 못한 경우.
    """
    if _get_chat_backend() is None:
        return _fallback_profile(answers)

    user_prompt = _build_extract_user_prompt(answers)
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_EXTRACT_ATTEMPTS + 1):
        try:
            raw_text = _call_llm(user_prompt, max_tokens=600, system_prompt=EXTRACT_SYSTEM_PROMPT)
            # 모델이 인사말/설명/코드블록 등 JSON 앞뒤로 군더더기를 붙이는 경우를 대비해
            # 첫 '{'~마지막 '}' 구간만 잘라낸 뒤 파싱한다.
            json_text = _extract_json_object(raw_text)
            profile = json.loads(json_text)
            _validate_profile(profile)
            return profile
        except Exception as exc:
            last_error = exc
            logger.warning(f"[extract_profile] 시도 {attempt}/{MAX_EXTRACT_ATTEMPTS} 실패: {exc}")

    raise RuntimeError(f"extract_profile 실패: 최대 재시도({MAX_EXTRACT_ATTEMPTS}회) 초과. 마지막 오류: {last_error}")
