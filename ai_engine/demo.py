"""
ai_engine/demo.py
===================
목데이터 기반 데모 실행부.

matching_engine.py의 generate_mock_users와 if __name__ == "__main__": 데모를 동작 변경
없이 그대로 옮겼고, 새로 추가된 리뷰 기반 프로필 보정(feedback.py)과 식당 확정
(restaurant.py) 흐름을 확인하는 데모 섹션을 덧붙였다.

실행: python -m ai_engine.demo
API 키가 전혀 없어도(LLM_BASE_URL/ANTHROPIC_API_KEY 모두 미설정) 모든 섹션이 규칙 기반
폴백으로 끝까지 동작한다.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from .feedback import (
    check_discrepancy_threshold,
    clip_delta,
    compute_discrepancy,
    run_feedback_pipeline,
    update_behavior_vector,
)
from .matching import run_matching_pipeline
from .restaurant import resolve_restaurant
from .schemas import MatchGroup, UserProfile


# =====================================================================
# 1. 테스트용 목데이터
# =====================================================================

def generate_mock_users(n: int = 8) -> List[Dict[str, Any]]:
    """
    무엇을: LLM 호출 없이, 이미 extract_profile()을 거쳤다고 가정한 가상 사용자 n명을
    하드코딩된 데이터로 생성한다.

    왜: 데모/테스트 단계에서 매 실행마다 API 호출 비용과 지연을 감수할 필요는 없다.
    매칭 로직(personality_similarity, diversity_score, form_groups)만 독립적으로 검증하려면
    입력 프로필은 고정되어 있는 편이 재현성과 디버깅에 유리하다.

    왜 이렇게 다양하게 구성했는가: 전공/학교/성격이 겹치는 쌍과 겹치지 않는 쌍을 고루 섞어야
    diversity_score와 personality_similarity가 실제로 서로 다른 값을 만들어내는지, 그리고
    form_groups가 그 차이를 반영해 그룹을 나누는지 확인할 수 있다.

    university/location 설계 근거: 이 서비스는 원래 전국 대학생 대상이지만, 데모 목데이터는
    국민대/숭실대/순천향대 세 대학 연합 해커톤 현장에 실제로 모인 상황을 가정한다 - 출신
    학교는 다양성 매칭(diversity_score)에 쓰이고, 물리적 위치는 다들 같은 행사장에 있다는
    전제로 식당 추천이 실제로 작동하도록 국민대학교 미래관(37.6103, 126.9974) 인근 반경
    1km 이내에 좌표를 ±0.003 정도씩만 흩어서 모아둔다. budget 필드는 restaurant.py 데모
    (resolve_restaurant)를 위해 추가로 넣어둔 것으로, 기존 매칭 로직
    (personality_similarity/diversity_score/pair_score)은 location/budget 키를 전혀
    참조하지 않으므로 기존 동작에는 영향이 없다.
    """
    sample_pool = [
        {
            "name": "김하늘", "university": "국민대", "major": "소프트웨어학부",
            "openness": 4, "conscientiousness": 3, "extraversion": 4, "agreeableness": 4, "neuroticism": 2,
            "interest_tags": ["보드게임", "인공지능", "캠핑"],
            "location": (37.6124, 126.9959), "budget": (8000, 15000),
        },
        {
            "name": "이도윤", "university": "숭실대", "major": "경영학부",
            "openness": 3, "conscientiousness": 4, "extraversion": 5, "agreeableness": 3, "neuroticism": 2,
            "interest_tags": ["창업", "농구", "여행"],
            "location": (37.6085, 126.9998), "budget": (10000, 20000),
        },
        {
            "name": "박서연", "university": "순천향대", "major": "컴퓨터소프트웨어공학과",
            "openness": 4, "conscientiousness": 3, "extraversion": 3, "agreeableness": 4, "neuroticism": 3,
            "interest_tags": ["인공지능", "영화", "카페투어"],
            "location": (37.6112, 127.0001), "budget": (9000, 16000),
        },
        {
            "name": "최민준", "university": "국민대", "major": "심리학과",
            "openness": 5, "conscientiousness": 2, "extraversion": 2, "agreeableness": 5, "neuroticism": 3,
            "interest_tags": ["독서", "심리상담", "글쓰기"],
            "location": (37.6077, 126.9966), "budget": (7000, 13000),
        },
        {
            "name": "정유진", "university": "국민대", "major": "시각디자인학과",
            "openness": 5, "conscientiousness": 3, "extraversion": 4, "agreeableness": 4, "neuroticism": 2,
            "interest_tags": ["전시회", "카페투어", "사진"],
            "location": (37.6116, 126.9952), "budget": (8000, 18000),
        },
        {
            "name": "한지훈", "university": "숭실대", "major": "기계공학부",
            "openness": 2, "conscientiousness": 5, "extraversion": 2, "agreeableness": 3, "neuroticism": 3,
            "interest_tags": ["자동차", "캠핑", "등산"],
            "location": (37.6096, 126.9985), "budget": (10000, 15000),
        },
        {
            "name": "오세연", "university": "순천향대", "major": "간호학과",
            "openness": 3, "conscientiousness": 5, "extraversion": 3, "agreeableness": 5, "neuroticism": 2,
            "interest_tags": ["운동", "요리", "여행"],
            "location": (37.6131, 126.9980), "budget": (9000, 14000),
        },
        {
            "name": "강도현", "university": "숭실대", "major": "전자정보공학부",
            "openness": 3, "conscientiousness": 4, "extraversion": 3, "agreeableness": 3, "neuroticism": 4,
            "interest_tags": ["게임", "인공지능", "농구"],
            "location": (37.6089, 126.9949), "budget": (8000, 15000),
        },
    ]
    return sample_pool[:n]


# 국민대학교 미래관(37.6103, 126.9974) 반경 1.5km 이내, 정릉·길음·성북 인근 제휴 식당 14곳.
# generate_mock_users()의 모든 위치가 이 중심점 인근 1km 반경에 모여 있으므로, 어떤 그룹
# 조합이 와도 resolve_restaurant()가 조건 완화 없이(또는 최대 1단계 완화로) 식당을 찾을 수
# 있도록 가격대(7000~25000원)와 방향을 고루 분산시켰다.
MOCK_RESTAURANT_DB: List[Dict[str, Any]] = [
    {"name": "정릉동 큰집순두부", "lat": 37.6122, "lng": 126.9968, "price_per_person": 9000},
    {"name": "미래관 앞 김밥천국", "lat": 37.6106, "lng": 126.9977, "price_per_person": 7000},
    {"name": "정릉시장 왕돈까스", "lat": 37.6095, "lng": 126.9958, "price_per_person": 11000},
    {"name": "북악산 스카이카페", "lat": 37.6132, "lng": 126.9990, "price_per_person": 8500},
    {"name": "정릉천 떡볶이포차", "lat": 37.6115, "lng": 126.9950, "price_per_person": 7500},
    {"name": "국민대 후문 화로구이", "lat": 37.6085, "lng": 127.0002, "price_per_person": 22000},
    {"name": "정릉동 이자카야 하나", "lat": 37.6088, "lng": 126.9935, "price_per_person": 20000},
    {"name": "성북 국밥거리", "lat": 37.6070, "lng": 126.9965, "price_per_person": 9500},
    {"name": "정릉 파스타공방", "lat": 37.6078, "lng": 126.9945, "price_per_person": 15000},
    {"name": "아리랑고개 초밥마을", "lat": 37.6090, "lng": 126.9995, "price_per_person": 24000},
    {"name": "정릉동 스타벅스", "lat": 37.6118, "lng": 126.9990, "price_per_person": 7800},
    {"name": "길음동 닭한마리", "lat": 37.6065, "lng": 126.9958, "price_per_person": 16000},
    {"name": "정릉 온기 백반집", "lat": 37.6100, "lng": 126.9945, "price_per_person": 8000},
    {"name": "미래관 카페테리아2호점", "lat": 37.6112, "lng": 126.9985, "price_per_person": 7200},
]


# =====================================================================
# 2. 매칭 데모 (기존 matching_engine.py __main__ 그대로)
# =====================================================================

def _run_matching_demo() -> List[Dict[str, Any]]:
    users = generate_mock_users(8)

    print("=" * 70)
    print(f"AI 기반 랜덤 식사 매칭 데모 - 대기 인원 {len(users)}명")
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("LLM_BASE_URL"):
        print("(LLM 백엔드가 설정되어 있지 않아 규칙 기반 폴백으로 동작합니다.)")
    print("=" * 70)

    match_history: Dict[Any, int] = {}
    # run_matching_pipeline은 그룹별 generate_match_reason/generate_icebreakers를
    # ThreadPoolExecutor(max_workers=8)로 병렬 실행한다. API 키가 없는 지금은 규칙 기반
    # 폴백이라 어차피 순식간에 끝나지만, 실제 LLM을 붙였을 때 이 시간이 "그룹 수 x 2회
    # LLM 왕복" 대신 "가장 느린 호출 1회"에 수렴하는지 확인할 수 있도록 소요 시간을 재둔다.
    start = time.perf_counter()
    pipeline_results = run_matching_pipeline(users, group_size=4, match_history=match_history)
    elapsed = time.perf_counter() - start
    print(f"[타이밍] run_matching_pipeline 소요 시간: {elapsed:.3f}초 (그룹 {len(pipeline_results)}개, LLM 호출 최대 {len(pipeline_results) * 2}회, ThreadPoolExecutor(max_workers=8)로 병렬 실행)")

    for idx, group_result in enumerate(pipeline_results, start=1):
        print(f"\n[그룹 {idx}]")
        for member in group_result["members"]:
            interests = ", ".join(member.get("interest_tags", []))
            print(f"  - {member['name']} ({member['university']} / {member['major']}) | 관심사: {interests}")
        print(f"  매칭 이유: {group_result['match_reason']}")
        print("  아이스브레이커:")
        for q in group_result["icebreakers"]:
            print(f"    · {q}")

    print(f"\n[매칭 이력 테이블] {len(match_history)}개 쌍 기록됨 (다음 라운드 반복 매칭 페널티에 사용)")
    return pipeline_results


# =====================================================================
# 3. 식당 확정 데모 (restaurant.py)
# =====================================================================

def _run_restaurant_demo(pipeline_results: List[Dict[str, Any]]) -> None:
    if not pipeline_results:
        print("\n(그룹이 하나도 만들어지지 않아 식당 확정 데모는 건너뜁니다.)")
        return

    print("\n" + "=" * 70)
    print("식당 확정 데모 (그룹 1)")
    print("=" * 70)

    group = MatchGroup(**pipeline_results[0])
    result = resolve_restaurant(group, MOCK_RESTAURANT_DB)
    print(f"  중간지점: {result['midpoint']}")
    print(f"  예산 교집합: {result['budget_range']}")
    print(f"  조건 완화 횟수: {result['relaxed_steps']}")
    print(f"  결과: {result['message']}")


# =====================================================================
# 4. 리뷰 기반 프로필 보정 데모 (feedback.py)
# =====================================================================

def _run_feedback_demo() -> None:
    print("\n" + "=" * 70)
    print("리뷰 기반 프로필 보정 데모")
    print("=" * 70)

    base_user = generate_mock_users(1)[0]
    profile = UserProfile.from_extracted_profile(
        base_user,
        name=base_user["name"],
        university=base_user["university"],
        major=base_user["major"],
    ).to_dict()
    print(f"  초기 self_report_vector: {profile['self_report_vector']}")
    print(f"  초기 behavior_corrected_vector: {profile['behavior_corrected_vector']}")
    print(f"  초기 diversity_beta: {profile['diversity_beta']}")

    # --- 케이스 1: 정상 리뷰 제출 (24시간 이내, 실패 이력 없음) ---------------
    raw_review_answers = {
        "r1": "낯선 전공 사람들이랑 얘기하는 게 생각보다 재밌었고, 다음엔 더 새로운 모임에도 나가보고 싶어요.",
        "r2": "그래도 낯가림 때문에 처음엔 많이 긴장했어요.",
        "r3": "다들 친절해서 편하게 얘기할 수 있었어요.",
        "r4_choice": "낯선 배경의 사람들과 얘기하는 게 새롭고 재밌었어요",
    }
    result_1 = run_feedback_pipeline(raw_review_answers, profile, hours_since_meal=5.0, failure_count=0)
    print("\n  [케이스 1] 정상 제출 (식사 후 5시간, 실패 0회)")
    print(f"    deltas: {result_1['deltas']}")
    print(f"    갱신된 behavior_corrected_vector: {result_1['profile']['behavior_corrected_vector']}")
    print(f"    갱신된 diversity_beta: {result_1['profile']['diversity_beta']}")
    print(f"    profile_shift_event: {result_1['profile_shift_event']}")

    # --- 케이스 2: 24시간 초과 (타임아웃 스킵) --------------------------------
    result_2 = run_feedback_pipeline(raw_review_answers, result_1["profile"], hours_since_meal=30.0, failure_count=0)
    print("\n  [케이스 2] 타임아웃 (식사 후 30시간)")
    print(f"    deltas: {result_2['deltas']} (None이어야 정상 - 델타 0 처리)")
    print(f"    profile_shift_event: {result_2['profile_shift_event']}")

    # --- 케이스 3: 반복 실패 2회 (스킵) ---------------------------------------
    result_3 = run_feedback_pipeline(raw_review_answers, result_1["profile"], hours_since_meal=5.0, failure_count=2)
    print("\n  [케이스 3] 파싱/검증 2회 실패 후 재시도")
    print(f"    deltas: {result_3['deltas']} (None이어야 정상 - 델타 0 처리)")

    # --- 케이스 4: 임계값 초과(profile_shift_event=True)가 실제로 트리거되는지 확인 ---
    # LLM 없이도(폴백 시 delta=0) 로직 자체를 검증하기 위해, 동일 축에 큰 델타가
    # 여러 라운드 누적된 상황을 직접 구성해 check_discrepancy_threshold까지 통과시켜본다.
    drifted_profile = dict(profile)
    drifted_profile["behavior_corrected_vector"] = dict(profile["self_report_vector"])
    accumulated = 0.0
    for _ in range(6):  # 학습률 0.5 * 델타 1.0 = 0.5씩, 누적 상한 1.5까지 여러 번 반영
        clipped = clip_delta(1.0, current_accumulated=accumulated)
        drifted_profile = update_behavior_vector(drifted_profile, {"openness": clipped, "conscientiousness": 0.0, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 0.0})
        accumulated += clipped

    discrepancy = compute_discrepancy(drifted_profile["self_report_vector"], drifted_profile["behavior_corrected_vector"])
    shift_event = check_discrepancy_threshold(discrepancy)
    print("\n  [케이스 4] openness에 큰 델타를 여러 라운드 누적시켜 임계값(1.0) 초과 유도")
    print(f"    누적 후 openness discrepancy: {discrepancy['openness']:.2f} (accumulated_cap=1.5로 상한)")
    print(f"    profile_shift_event: {shift_event} (True여야 정상)")


# =====================================================================
# 5. 진입점
# =====================================================================

def main() -> None:
    pipeline_results = _run_matching_demo()
    _run_restaurant_demo(pipeline_results)
    _run_feedback_demo()
    print("\n" + "=" * 70)
    print("데모 종료.")


if __name__ == "__main__":
    main()
