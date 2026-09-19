"""
ai_engine/server.py
=====================
mind_table AI 매칭 엔진을 감싸는 FastAPI 서버.

실행:
    uvicorn ai_engine.server:app --reload --port 8000

문서: 서버 실행 후 http://localhost:8000/docs 에서 자동 생성된 OpenAPI 문서(Swagger UI)를
확인할 수 있다.

설계 원칙: 이 파일은 ai_engine의 기존 로직(profile/matching/restaurant/feedback)을 절대
수정하지 않고, 그 위에 "요청 파싱 -> 기존 함수 호출 -> 응답 변환/예외 매핑"만 담당하는
얇은 계층이다. 실제 매칭/추출/보정 알고리즘은 전부 각 모듈에 그대로 남아 있다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import api_models, feedback, matching, profile as profile_module, restaurant, schemas, store

logger = logging.getLogger("ai_engine.server")

app = FastAPI(title="mind_table AI Engine API", version="0.1.0")

# CORS: 해커톤 데모용으로 전체 허용(allow_origins=["*"]). 프론트가 아직 어느 origin/port
# 에서 뜰지 정해지지 않은 개발 초기 단계의 마찰을 없애기 위한 설정이며, 프로덕션에서는
# 반드시 실제 프론트 도메인으로 origins를 좁혀야 한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    """헬스체크: 서버 프로세스가 떠 있고 요청을 받을 수 있는지만 빠르게 확인한다."""
    return {"status": "ok"}


# =====================================================================
# POST /profile/extract
# =====================================================================

@app.post("/profile/extract", response_model=api_models.UserProfile)
def extract_profile_endpoint(answers: api_models.OnboardingAnswers) -> api_models.UserProfile:
    """
    무엇을: 온보딩 답변 4개(q1~q3 자유서술, q4_choice 강제선택)를 받아
    profile.extract_profile()로 Big Five 5축 + 관심사 키워드를 추출하고,
    schemas.UserProfile.from_extracted_profile()로 감싸 반환한다.

    왜 500과 400을 구분하는가: extract_profile()은 LLM 응답이 스키마를 계속 벗어나면
    (최대 2회 재시도 후에도) RuntimeError를 던진다. 이건 우리 서버나 사용자 입력의 문제가
    아니라 LLM 게이트웨이 쪽 문제이므로 500(서버 오류)으로 매핑한다. 그 외의 예외(예상치
    못한 값 형태 등)는 400으로 구분해, 클라이언트가 "내가 보낸 데이터가 잘못됐다"는 걸
    알 수 있게 한다.
    """
    try:
        extracted = profile_module.extract_profile(answers.model_dump())
    except RuntimeError as exc:
        logger.error(f"[/profile/extract] extract_profile 재시도 초과: {exc}")
        raise HTTPException(status_code=500, detail=f"프로필 추출에 반복 실패했습니다: {exc}")
    except Exception as exc:
        logger.error(f"[/profile/extract] 예기치 못한 오류: {exc}")
        raise HTTPException(status_code=400, detail=f"입력을 처리할 수 없습니다: {exc}")

    user_profile = schemas.UserProfile.from_extracted_profile(extracted)
    return api_models.UserProfile.from_dataclass(user_profile)


# =====================================================================
# POST /match/run
# =====================================================================

class MatchGroupResult(BaseModel):
    members: List[Dict[str, Any]]
    match_reason: str
    icebreakers: List[str]


class MatchRunResponse(BaseModel):
    groups: List[MatchGroupResult]
    leftover: List[Dict[str, Any]]


def _candidate_to_matching_dict(candidate: api_models.MatchCandidate) -> Dict[str, Any]:
    """
    무엇을: API로 받은 MatchCandidate(자기보고/행동보정 벡터가 분리된 UserProfile 스타일
    구조)를, matching.py의 personality_similarity/diversity_score/pair_score가 그대로
    기대하는 "평탄한" dict(openness/conscientiousness/.../neuroticism이 최상위 키)로
    변환한다.

    왜 여기서 변환하는가: matching.py는 이미 실제 게이트웨이로 검증이 끝난 로직이라 절대
    수정하지 않기로 했다. 대신 API 계층에서 UserProfile 스타일 중첩 구조를 matching.py가
    기대하는 평탄한 구조로 변환해주는 어댑터 역할을 이 함수가 맡는다.

    왜 behavior_corrected_vector를 우선하는가: 리뷰 피드백으로 보정된 벡터가 있다면
    그게 "실제 식사 자리에서 드러나는 성향"에 더 가깝다(feedback.py 참고). 있으면 그것을
    매칭 기준으로 쓰고, 없으면(아직 리뷰가 없는 신규 유저) self_report_vector로 폴백한다.
    """
    vector = candidate.behavior_corrected_vector or candidate.self_report_vector
    flat: Dict[str, Any] = {
        "name": candidate.name,
        "university": candidate.university,
        "major": candidate.major,
        "interest_tags": list(candidate.interest_tags),
    }
    if candidate.user_id:
        flat["user_id"] = candidate.user_id

    for trait in schemas.BIG_FIVE_TRAITS:
        if trait not in vector:
            raise ValueError(f"'{candidate.name}'의 성격 벡터에 '{trait}' 축이 없습니다.")
        flat[trait] = vector[trait]
    return flat


def _strip_internal_keys(user: Dict[str, Any]) -> Dict[str, Any]:
    """matching.pair_score가 캐싱용으로 붙인 내부 키(_interest_embedding 등)를 응답에서 제거한다."""
    return {k: v for k, v in user.items() if not k.startswith("_")}


@app.post("/match/run", response_model=MatchRunResponse)
def run_match_endpoint(request: api_models.MatchRunRequest) -> MatchRunResponse:
    """
    무엇을: 후보 사용자 리스트를 받아 matching.run_matching_pipeline()으로 그룹을
    구성한다. store.get_match_history()로 과거 매칭 이력을 불러와 반복 매칭 페널티에
    반영하고, 실행 후 갱신된 이력을 store.save_match_history()로 다시 저장한다.

    왜 leftover를 직접 계산하는가: run_matching_pipeline()은 이월 인원을 로그로만 남기고
    반환값에는 포함하지 않는다(matching.py를 수정하지 않기로 했으므로 그대로 둔다). 대신
    matching.form_groups()가 입력 dict 객체를 그대로 재사용한다는 점(복사하지 않음)을
    이용해, 결과 그룹들의 members에 포함되지 않은 입력 사용자를 파이썬 객체 identity로
    걸러내 leftover를 재구성한다 - form_groups를 다시 호출하지 않으므로 임베딩 API가
    중복 호출되지 않는다.
    """
    try:
        flat_users = [_candidate_to_matching_dict(c) for c in request.candidates]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    match_history = store.get_match_history()
    try:
        results = matching.run_matching_pipeline(
            flat_users, group_size=request.group_size, match_history=match_history
        )
    except Exception as exc:
        logger.error(f"[/match/run] run_matching_pipeline 실패: {exc}")
        raise HTTPException(status_code=500, detail=f"매칭 실행 중 오류가 발생했습니다: {exc}")

    store.save_match_history(match_history)

    assigned_ids = {id(member) for group in results for member in group["members"]}
    leftover = [_strip_internal_keys(u) for u in flat_users if id(u) not in assigned_ids]

    groups = [
        MatchGroupResult(
            members=[_strip_internal_keys(m) for m in g["members"]],
            match_reason=g["match_reason"],
            icebreakers=g["icebreakers"],
        )
        for g in results
    ]
    return MatchRunResponse(groups=groups, leftover=leftover)


# =====================================================================
# POST /restaurant/resolve
# =====================================================================

@app.post("/restaurant/resolve")
def resolve_restaurant_endpoint(group: api_models.MatchGroup) -> Dict[str, Any]:
    """
    무엇을: 확정된 그룹(members 각자에 location/budget 포함)을 받아
    store.get_restaurant_db()의 제휴 식당 중 조건(중간지점 반경, 예산 교집합)에 맞는 곳을
    restaurant.resolve_restaurant()로 확정한다.

    왜 ValueError를 400으로 따로 잡는가: resolve_restaurant()는 그룹 구성원 전원에게
    위치 정보가 하나도 없으면 ValueError를 던진다. 이건 서버 오류가 아니라 클라이언트가
    location 없는 그룹을 보낸 입력 문제이므로 400으로 구분한다.
    """
    try:
        match_group = group.to_dataclass()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"그룹 데이터가 올바르지 않습니다: {exc}")

    restaurant_db = store.get_restaurant_db()
    try:
        result = restaurant.resolve_restaurant(match_group, restaurant_db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"[/restaurant/resolve] resolve_restaurant 실패: {exc}")
        raise HTTPException(status_code=500, detail=f"식당 확정 중 오류가 발생했습니다: {exc}")

    return result


# =====================================================================
# POST /review/submit
# =====================================================================

class ReviewSubmitRequest(BaseModel):
    r1: str
    r2: str
    r3: str
    r4_choice: str
    profile: api_models.UserProfile
    hours_since_meal: float
    failure_count: int = 0


class ReviewSubmitResponse(BaseModel):
    profile: api_models.UserProfile
    profile_shift_event: bool
    deltas: Optional[Dict[str, float]] = None


@app.post("/review/submit", response_model=ReviewSubmitResponse)
def submit_review_endpoint(request: ReviewSubmitRequest) -> ReviewSubmitResponse:
    """
    무엇을: 리뷰 원본 답변(r1~r4_choice) + 현재 profile을 받아
    feedback.run_feedback_pipeline()을 실행하고, 갱신된 profile/임계 초과 여부
    (profile_shift_event)/축별 델타를 반환한다.

    왜 KeyError를 400으로 따로 잡는가: run_feedback_pipeline()은 profile 안에
    self_report_vector가 없으면 KeyError를 던진다. 이는 클라이언트가 UserProfile 형태가
    아닌 profile을 보낸 입력 문제이므로 400으로 구분한다. (extract_delta 자체의 LLM 실패는
    feedback.py 내부에서 이미 델타 0으로 폴백 처리되어 여기까지 예외가 올라오지 않는다.)
    """
    raw_answers = {"r1": request.r1, "r2": request.r2, "r3": request.r3, "r4_choice": request.r4_choice}
    profile_dict = request.profile.model_dump()

    try:
        result = feedback.run_feedback_pipeline(
            raw_answers, profile_dict, request.hours_since_meal, request.failure_count
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"profile에 필수 필드가 없습니다: {exc}")
    except Exception as exc:
        logger.error(f"[/review/submit] run_feedback_pipeline 실패: {exc}")
        raise HTTPException(status_code=500, detail=f"리뷰 처리 중 오류가 발생했습니다: {exc}")

    return ReviewSubmitResponse(
        profile=api_models.UserProfile.model_validate(result["profile"]),
        profile_shift_event=result["profile_shift_event"],
        deltas=result["deltas"],
    )
