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
# 통합 결정(INTEGRATION_REPORT.md): 김주환 브랜치는 이 축을 O/C/E/A/N 단일문자로 표기했지만,
# API 경계 가독성을 위해 jaebin의 영문 풀네임으로 전부 통일했다 - 어댑터 계층 없이 모든
# 모듈(profile/matching/feedback)이 이 리스트 하나만 공유한다.
BIG_FIVE_TRAITS: List[str] = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

# 관심사 유사도 계산 전용 고정 태그 공간(김주환 브랜치 src/schema.py TAGS 이식).
# 자유 키워드(interest_tags)만으로 유사도를 계산하면 "클라이밍"과 "등산"이 남남이 되고,
# 임베딩 API가 없는 게이트웨이 환경에서는 유사도가 0으로 붕괴한다. 그래서 12개 고정
# 태그로 투영한 공간에서 코사인 유사도를 계산한다 - 유사도 계산의 실제 공간은 이 TAGS이고,
# interest_tags는 카드/아이스브레이커 표시용 자유 키워드로 분리 유지한다.
TAGS: List[str] = [
    "운동", "여행", "요리", "음악", "영상", "독서",
    "게임", "기술", "학술", "창작", "봉사", "재테크",
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
        behavior_corrected_vector: self_report_vector에 **가산되는 오프셋**(축마다 0.0에서
                                    시작, ±BEHAVIOR_CLIP=1.5로 클립). "실제 행동에서 드러난
                                    성향이 자기보고 대비 어느 방향으로 얼마나 벗어났는가"를
                                    나타낸다. 매칭에 쓰는 "현재 값"은 항상
                                    clamp(self_report + behavior, 1, 5)로 계산한다
                                    (feedback.py의 EMA 갱신식, 통합 이후 오프셋 방식으로 확정).
        diversity_beta            : 이 사용자의 개인화된 다양성 선호도(0.1~0.9, 기본 0.5).
                                    리뷰의 강제선택 문항에 따라 update_diversity_beta()가
                                    조정한다.
        name/university/major/interest_tags : matching.py의 pair_score 등 기존 dict 기반
                                    매칭 로직과 호환되도록 함께 들고 있는 부가 정보.
        tags                      : TAGS(고정 12종) 중 이 사용자에게 분류된 태그(1~3개).
                                    matching.py의 유사도 계산은 interest_tags가 아니라
                                    이 tags + interest_weights로 이루어진다(김주환 브랜치
                                    이식, 임베딩 없는 게이트웨이에서도 유사도가 안 붕괴).
        interest_weights          : tags별 가중치. 온보딩 시 1.0으로 초기화되고, 리뷰에서
                                    "실제로 대화가 통한 주제"가 나오면 feedback.py가 올리고,
                                    죽은 주제는 내린다 - 매칭이 참조하는 유사도 공간이
                                    회차를 거듭할수록 실제 경험을 반영하도록 학습된다.
        confidence                : extract_profile()이 매긴 이 추출 결과의 신뢰도(0~1).
                                    답변이 짧거나 모호하면 낮다.
    """

    self_report_vector: Dict[str, float]
    behavior_corrected_vector: Dict[str, float]
    diversity_beta: float = 0.5
    name: Optional[str] = None
    university: Optional[str] = None
    major: Optional[str] = None
    interest_tags: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    interest_weights: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

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
        무엇을: profile.extract_profile()이 반환한 원시 dict(성격 5축 + interest_tags +
        tags + confidence)를 UserProfile로 감싼다. behavior_corrected_vector는 축마다
        0.0(오프셋 없음)으로 초기화한다 - 아직 리뷰가 없으므로 자기보고와 "현재 값"이
        완전히 같다. tags는 전부 가중치 1.0으로 초기화되고, 이후 리뷰가 쌓이면서
        interest_weights가 움직인다.
        """
        self_report_vector = {trait: float(extracted[trait]) for trait in BIG_FIVE_TRAITS}
        tags = [t for t in extracted.get("tags", []) if t in TAGS]
        return cls(
            self_report_vector=self_report_vector,
            behavior_corrected_vector={trait: 0.0 for trait in BIG_FIVE_TRAITS},
            diversity_beta=diversity_beta,
            name=name,
            university=university,
            major=major,
            interest_tags=list(extracted.get("interest_tags", [])),
            tags=tags,
            interest_weights={t: 1.0 for t in tags},
            confidence=float(extracted.get("confidence", 0.0) or 0.0),
        )


@dataclass
class MatchGroup:
    """
    무엇을: run_matching_pipeline()이 만들어내는 그룹 하나의 결과를 감싸는 구조.
    필드 이름이 run_matching_pipeline의 반환 dict 키와 정확히 일치하므로,
    MatchGroup(**group_dict)로 바로 변환해 restaurant.py 등 다음 단계로 넘길 수 있다.

    icebreaker_targets/overlap(통합 이후 추가): 김주환 브랜치 cards.py 이식으로
    matching.py가 함께 생성하게 된 필드 - 각 아이스브레이커 질문에 "누가 답할 수
    있는지"(targets)와 그룹이 공유하는 관심사(overlap)를 프론트엔드 카드 UI가
    바로 쓸 수 있게 노출한다.
    """

    members: List[Dict[str, Any]]
    match_reason: Optional[str] = None
    icebreakers: List[str] = field(default_factory=list)
    icebreaker_targets: List[List[str]] = field(default_factory=list)
    overlap: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
