"""
ai_engine/schemas.py
======================
패키지 전체에서 공유하는 데이터 구조 정의.

기존 matching_engine.py의 함수들(personality_similarity, diversity_score, pair_score 등)은
동작을 바꾸지 않기 위해 그대로 dict 기반으로 남겨뒀다. 여기 정의된 dataclass들은 주로
(1) 새로 추가되는 리뷰 기반 보정 시스템(UserProfile, ReviewAnswers, DeltaResult)과
(2) 백엔드 API 계층과의 타입 안전한 경계(OnboardingAnswers, MatchGroup)를 위한 것이다.

dict와 자유롭게 오갈 수 있도록 각 dataclass는 최소한의 변환 메서드를 제공한다
(dataclasses.asdict()로도 항상 변환 가능하다).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# Big Five(OCEAN) 성격 5요인 축 이름. profile.py/matching.py/feedback.py가 공통으로 참조한다.
BIG_FIVE_TRAITS: List[str] = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


@dataclass
class OnboardingAnswers:
    """
    무엇을: 온보딩 설문 4문항(자유서술 3개 + 강제선택 1개)의 타입 안전한 표현.
    왜: extract_profile()은 하위 호환을 위해 dict 시그니처를 그대로 유지하지만, 백엔드
    API 계층(요청 바디 검증 등)에서는 이 dataclass로 먼저 파싱해두면 필드 오타나 누락을
    타입 체커/직렬화 라이브러리 수준에서 미리 잡을 수 있다.
    """

    q1: str
    q2: str
    q3: str
    q4_choice: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ReviewAnswers:
    """
    무엇을: 식사 후 리뷰 설문(자유서술 3개 + 강제선택 1개)의 분리된 표현.
    왜: extract_delta는 자유서술 텍스트만 필요하고, update_diversity_beta는 강제선택
    답변만 필요하다 - 두 흐름이 쓰는 재료가 다르므로 애초에 분리된 구조로 들고 있는 편이
    이후 파이프라인 코드를 단순하게 만든다.
    """

    free_text: List[str]  # 자유서술 답변 3개 (r1, r2, r3)
    forced_choice: str  # 다양성 선호도 관련 강제선택 답변 (r4_choice)

    def to_dict(self) -> Dict[str, Any]:
        return {"free_text": list(self.free_text), "forced_choice": self.forced_choice}


@dataclass
class DeltaResult:
    """
    무엇을: Big Five 한 축에 대해 extract_delta가 추정한 보정값(delta)과 그 근거가 된
    리뷰 원문 인용구(quote)를 함께 담는 단위 구조.
    왜: delta 숫자만 남기면 "왜 이 축이 이렇게 움직였는지"를 나중에 설명할 수 없다.
    quote를 함께 보관해두면(단, update_behavior_vector 처리 후에는 delta만 프로필에
    남고 quote/원문은 폐기된다 - feedback.py의 개인정보 최소 보관 원칙 참고) 디버깅이나
    사용자에게 "왜 이렇게 바뀌었는지" 설명할 때 근거로 쓸 수 있다.
    """

    delta: float  # -1.0 ~ +1.0
    quote: str  # 근거가 된 리뷰 원문 발췌 (없으면 빈 문자열)

    def to_dict(self) -> Dict[str, Any]:
        return {"delta": self.delta, "quote": self.quote}


@dataclass
class UserProfile:
    """
    무엇을: 한 사용자의 전체 성격/취향 프로필. 온보딩 직후 생성되어, 이후 리뷰가 쌓일
    때마다 behavior_corrected_vector와 diversity_beta만 갱신된다.

    필드:
        self_report_vector       : extract_profile() 직후 확정되는, 온보딩 자기보고 기반
                                    Big Five 5축 점수(1~5). **온보딩 이후 절대 변하지 않는다**
                                    (profile.py의 extract_profile 참고). "이 사람이 스스로를
                                    어떻게 인식하는가"의 고정 기준점 역할을 한다.
        behavior_corrected_vector: 리뷰 피드백(feedback.py)으로 라운드마다 조금씩 보정되는
                                    가변 벡터. 초기값은 self_report_vector의 복사본이며,
                                    "실제 식사 모임에서 드러난 행동 성향"을 반영해 표류한다.
        diversity_beta            : 이 사용자의 개인화된 다양성 선호도(0.1~0.9, 기본 0.5).
                                    리뷰의 강제선택 문항에 따라 update_diversity_beta()가
                                    조정한다.
        name/university/major/interest_tags : matching.py의 pair_score 등 기존 dict 기반
                                    매칭 로직과 호환되도록 함께 들고 있는 부가 정보.
    """

    self_report_vector: Dict[str, float]
    behavior_corrected_vector: Dict[str, float]
    diversity_beta: float = 0.5
    name: Optional[str] = None
    university: Optional[str] = None
    major: Optional[str] = None
    interest_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_extracted_profile(
        cls,
        extracted: Dict[str, Any],
        *,
        name: Optional[str] = None,
        university: Optional[str] = None,
        major: Optional[str] = None,
        diversity_beta: float = 0.5,
    ) -> "UserProfile":
        """
        무엇을: profile.extract_profile()이 반환한 원시 dict(성격 5축 + interest_tags)를
        UserProfile로 감싼다. self_report_vector와 behavior_corrected_vector를 이 시점에
        동일한 값으로 초기화한다(아직 리뷰가 없으므로 행동 보정치는 0).
        """
        self_report_vector = {trait: float(extracted[trait]) for trait in BIG_FIVE_TRAITS}
        return cls(
            self_report_vector=self_report_vector,
            behavior_corrected_vector=dict(self_report_vector),
            diversity_beta=diversity_beta,
            name=name,
            university=university,
            major=major,
            interest_tags=list(extracted.get("interest_tags", [])),
        )


@dataclass
class MatchGroup:
    """
    무엇을: run_matching_pipeline()이 만들어내는 그룹 하나의 결과를 감싸는 구조.
    필드 이름이 run_matching_pipeline의 반환 dict 키(members/match_reason/icebreakers)와
    정확히 일치하므로, MatchGroup(**group_dict)로 바로 변환해 restaurant.py 등 다음 단계로
    넘길 수 있다.
    """

    members: List[Dict[str, Any]]
    match_reason: Optional[str] = None
    icebreakers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
