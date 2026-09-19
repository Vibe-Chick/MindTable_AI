"""
ai_engine/api_models.py
=========================
FastAPI 요청/응답용 Pydantic 모델.

schemas.py의 dataclass(OnboardingAnswers, ReviewAnswers, UserProfile, MatchGroup)와
1:1 대응되는 Pydantic 버전을 정의한다. dataclass는 ai_engine 내부 로직(profile.py,
matching.py, restaurant.py, feedback.py)이 주고받는 "신뢰된" 내부 표현이고, 여기 Pydantic
모델은 HTTP 경계에서 들어오는 신뢰되지 않은 JSON을 검증하는 역할을 맡는다.

각 모델에 to_dataclass()/from_dataclass()를 달아둬서 서버 코드가 "요청 파싱 -> dataclass
변환 -> 기존 ai_engine 함수 호출 -> 결과를 다시 Pydantic으로" 흐름을 짧게 쓸 수 있게 했다.
dict로도 자유롭게 오갈 수 있도록 model_dump()/model_validate()가 그대로 dict를 받아들인다
(Pydantic v2 기본 동작).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import schemas


class OnboardingAnswers(BaseModel):
    """schemas.OnboardingAnswers와 1:1 대응. POST /profile/extract의 요청 바디로 쓰인다."""

    model_config = ConfigDict(from_attributes=True)

    q1: str
    q2: str
    q3: str
    q4_choice: str

    def to_dataclass(self) -> schemas.OnboardingAnswers:
        return schemas.OnboardingAnswers(**self.model_dump())

    @classmethod
    def from_dataclass(cls, obj: schemas.OnboardingAnswers) -> "OnboardingAnswers":
        return cls.model_validate(obj)


class ReviewAnswers(BaseModel):
    """
    schemas.ReviewAnswers와 1:1 대응(자유서술 3개 + 강제선택 1개가 이미 분리된 형태).
    참고: /review/submit 엔드포인트의 실제 요청 바디는 분리 이전 원본 답변(r1~r4_choice)을
    받는 ReviewSubmitRequest(server.py)이고, 그 안에서 feedback.parse_review()가 이
    ReviewAnswers 형태로 분리한다. 이 모델은 분리된 결과를 타입 안전하게 다루고 싶을 때
    (예: parse_review의 반환값을 검증) 쓰기 위해 1:1로 정의해 둔 것이다.
    """

    model_config = ConfigDict(from_attributes=True)

    free_text: List[str]
    forced_choice: str

    def to_dataclass(self) -> schemas.ReviewAnswers:
        return schemas.ReviewAnswers(**self.model_dump())

    @classmethod
    def from_dataclass(cls, obj: schemas.ReviewAnswers) -> "ReviewAnswers":
        return cls.model_validate(obj)


class UserProfile(BaseModel):
    """
    schemas.UserProfile과 1:1 대응. POST /profile/extract의 응답, POST /review/submit의
    요청(현재 profile)과 응답(갱신된 profile) 모두에 쓰인다.
    """

    model_config = ConfigDict(from_attributes=True)

    self_report_vector: Dict[str, float]
    behavior_corrected_vector: Dict[str, float]
    diversity_beta: float = 0.5
    name: Optional[str] = None
    university: Optional[str] = None
    major: Optional[str] = None
    interest_tags: List[str] = Field(default_factory=list)

    def to_dataclass(self) -> schemas.UserProfile:
        return schemas.UserProfile(**self.model_dump())

    @classmethod
    def from_dataclass(cls, obj: schemas.UserProfile) -> "UserProfile":
        return cls.model_validate(obj)


class MatchGroup(BaseModel):
    """
    schemas.MatchGroup과 1:1 대응. POST /restaurant/resolve의 요청 바디로 쓰인다
    (run_matching_pipeline이 만든 그룹 결과를 그대로 다시 넣는 형태).
    """

    model_config = ConfigDict(from_attributes=True)

    members: List[Dict[str, Any]]
    match_reason: Optional[str] = None
    icebreakers: List[str] = Field(default_factory=list)

    def to_dataclass(self) -> schemas.MatchGroup:
        return schemas.MatchGroup(**self.model_dump())

    @classmethod
    def from_dataclass(cls, obj: schemas.MatchGroup) -> "MatchGroup":
        return cls.model_validate(obj)


class MatchCandidate(BaseModel):
    """
    POST /match/run의 후보 사용자 1명. UserProfile과 비슷하지만 matching.py가 실제로
    필요로 하는 필드만 요구하도록(자기보고/행동보정 벡터 중 하나만 있어도 되게) 별도로
    정의했다. server.py의 _candidate_to_matching_dict()가 이 모델을
    matching.py 함수들이 기대하는 평탄한 dict(openness/…/neuroticism이 최상위 키)로
    변환한다.
    """

    model_config = ConfigDict(extra="allow")  # user_id 등 부가 식별자를 자유롭게 허용

    name: str
    university: Optional[str] = None
    major: Optional[str] = None
    interest_tags: List[str] = Field(default_factory=list)
    self_report_vector: Dict[str, float]
    behavior_corrected_vector: Optional[Dict[str, float]] = None
    user_id: Optional[str] = None


class MatchRunRequest(BaseModel):
    """POST /match/run 요청 바디."""

    candidates: List[MatchCandidate]
    group_size: int = 4
