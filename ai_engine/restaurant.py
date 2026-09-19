"""
ai_engine/restaurant.py
=========================
매칭이 확정된 그룹의 실제 식사 장소(제휴 식당)를 확정하는 로직.

매칭(matching.py)은 "누구와 누구를 묶을지"만 정하고, 실제로 어디서 밥을 먹을지는 여기서
결정한다. 그룹 구성원들의 위치/예산 정보를 받아 중간지점과 예산 교집합을 계산하고, 제휴
식당 DB에서 조건에 맞는 곳을 찾는다. 조건이 너무 빡빡해 결과가 없으면 단계적으로 완화해
재시도하고, 그래도 없으면 "확정 실패"를 명시적으로 반환한다(무리하게 아무 식당이나
끼워 맞추지 않는다).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from .schemas import MatchGroup

logger = logging.getLogger("ai_engine.restaurant")

DEFAULT_MAX_DISTANCE_KM = 1.5
DEFAULT_MAX_RELAX_STEPS = 3


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    무엇을: 두 위경도 좌표 사이의 실제 지표면 거리를 km 단위로 계산한다(haversine 공식).
    왜: 위도/경도는 평면 좌표가 아니므로 단순 좌표 차이(유클리드 거리)로는 거리를 왜곡한다.
    캠퍼스 반경 수 km 내 거리 비교에는 haversine이 충분히 정확하면서 외부 지도 API 호출
    없이 순수 계산만으로 가볍게 처리할 수 있다.
    """
    lat1, lon1 = a
    lat2, lon2 = b
    r_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r_km * math.asin(min(1.0, math.sqrt(h)))


def compute_midpoint(locations: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    무엇을: 그룹 구성원 위치들의 단순 평균(centroid)을 '중간지점'으로 계산한다.
    왜: 기하학적으로 가장 공정한 중앙점(예: Fermat point)을 구하는 건 계산이 복잡한데,
    실제 서비스에서는 애초에 반경 내 제휴 식당 후보 자체가 몇 곳으로 제한적이라, 단순
    평균으로도 "다 같이 이동하기 부담스럽지 않은 지점"을 근사하기에 충분하다.
    """
    lat_avg = sum(loc[0] for loc in locations) / len(locations)
    lng_avg = sum(loc[1] for loc in locations) / len(locations)
    return (lat_avg, lng_avg)


def compute_budget_intersection(budgets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """
    무엇을: 그룹 구성원별 (최소, 최대) 예산 구간의 교집합을 계산한다. 교집합이 없으면 None.
    왜: 한 명이라도 감당 못 하는 가격대의 식당을 고르면 만족도가 크게 떨어지므로, "모두가
    낼 수 있는" 구간을 먼저 좁혀서 후보를 필터링하는 것이 합리적이다.
    """
    lo = max(b[0] for b in budgets)
    hi = min(b[1] for b in budgets)
    if lo > hi:
        return None
    return (lo, hi)


def query_restaurants(
    restaurant_db: List[Dict[str, Any]],
    midpoint: Tuple[float, float],
    budget_range: Optional[Tuple[int, int]],
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
) -> List[Dict[str, Any]]:
    """
    무엇을: 제휴 식당 DB에서 중간지점 반경 max_distance_km 이내, 예산 교집합 내에 있는
    식당만 걸러 거리순으로 정렬해 반환한다. budget_range가 None이면(교집합 자체가 없던
    경우) 예산 필터는 건너뛰고 거리 조건만 적용한다 - 예산이 하나도 안 맞아도 장소 자체는
    제시할 수 있어야 운영진이 수동으로 조정할 여지가 생긴다.
    """
    candidates = []
    for restaurant in restaurant_db:
        loc = (restaurant["lat"], restaurant["lng"])
        distance = _haversine_km(midpoint, loc)
        if distance > max_distance_km:
            continue
        if budget_range is not None:
            price = restaurant.get("price_per_person")
            if price is not None and not (budget_range[0] <= price <= budget_range[1]):
                continue
        candidates.append({**restaurant, "_distance_km": distance})

    candidates.sort(key=lambda r: r["_distance_km"])
    return candidates


def resolve_restaurant(
    group: MatchGroup,
    restaurant_db: List[Dict[str, Any]],
    max_relax_steps: int = DEFAULT_MAX_RELAX_STEPS,
) -> Dict[str, Any]:
    """
    무엇을: 그룹 정보를 받아 중간지점/예산 교집합을 계산하고, 조건에 맞는 제휴 식당을
    확정한다. 후보가 없으면 거리 반경을 넓히고 예산 제약을 완화하며 최대 max_relax_steps번
    재시도한다.

    왜 조건 완화가 필요한가: 그룹 구성원이 캠퍼스 곳곳에 흩어져 있거나 예산대가 서로 크게
    다르면 엄격한 조건으로는 후보가 0개일 수 있다. 매칭 자체는 이미 확정된 상태이므로
    "완벽한 조건"보다 "실행 가능한 대안을 반드시 제시"하는 쪽이 서비스 신뢰도에 유리하다.
    단, 무한정 완화하지 않고 max_relax_steps로 상한을 둬서, 그래도 안 되면 운영진이
    수동으로 개입해야 한다는 신호(restaurant: None)를 명확히 준다.
    """
    members = group.members
    locations = [tuple(m["location"]) for m in members if m.get("location")]
    budgets = [tuple(m["budget"]) for m in members if m.get("budget")]

    if not locations:
        raise ValueError("그룹 구성원 중 위치 정보가 있는 사람이 없습니다.")

    midpoint = compute_midpoint(locations)
    budget_range = compute_budget_intersection(budgets) if budgets else None

    max_distance_km = DEFAULT_MAX_DISTANCE_KM
    relaxed_steps = 0
    candidates: List[Dict[str, Any]] = []

    while True:
        candidates = query_restaurants(restaurant_db, midpoint, budget_range, max_distance_km)
        if candidates or relaxed_steps >= max_relax_steps:
            break
        relaxed_steps += 1
        max_distance_km *= 1.5
        if budget_range is not None:
            budget_range = (max(0, int(budget_range[0] * 0.8)), int(budget_range[1] * 1.2))
        logger.warning(
            f"[resolve_restaurant] 조건에 맞는 식당이 없어 완화합니다 "
            f"(단계 {relaxed_steps}/{max_relax_steps}, 반경={max_distance_km:.1f}km, 예산={budget_range})"
        )

    if not candidates:
        return {
            "restaurant": None,
            "midpoint": midpoint,
            "budget_range": budget_range,
            "relaxed_steps": relaxed_steps,
            "message": "조건을 완화해도 확정 가능한 식당을 찾지 못했습니다. 운영자 확인이 필요합니다.",
        }

    chosen = candidates[0]
    return {
        "restaurant": chosen,
        "midpoint": midpoint,
        "budget_range": budget_range,
        "relaxed_steps": relaxed_steps,
        "message": f"{chosen.get('name', '식당')}(으)로 확정되었습니다 (중간지점에서 {chosen['_distance_km']:.2f}km).",
    }
