"""
ai_engine/store.py
=====================
상태 저장소 - 지금은 인메모리, 나중에 실제 DB로 교체 예정.

왜 함수 인터페이스로 감쌌는가:
    지금 당장은 DB가 없어서 모듈 전역 변수(_match_history, _restaurant_db)에 상태를 들고
    있지만, server.py나 다른 모듈이 이 전역 변수를 직접 import해서 건드리기 시작하면
    나중에 실제 DB(Redis, Postgres 등)로 바꿀 때 호출부를 전부 찾아 고쳐야 한다.
    get_match_history()/save_match_history()/get_restaurant_db() 세 함수 뒤에 상태를
    숨겨두면, 실제 DB로 옮길 때 이 파일 내부 구현만 바꾸면 되고(예: dict 읽기/쓰기를
    SELECT/UPSERT 쿼리로 교체) 호출부는 코드를 한 줄도 고칠 필요가 없다. 그래서 이 모듈
    밖으로는 전역 변수를 절대 직접 노출하지 않는다.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from .demo import MOCK_RESTAURANT_DB

# --- 모듈 프라이빗 상태 - 아래 함수들을 통해서만 접근한다 -------------------
_match_history: Dict[Tuple[str, str], int] = {}
_restaurant_db: List[Dict[str, Any]] = copy.deepcopy(MOCK_RESTAURANT_DB)


def get_match_history() -> Dict[Tuple[str, str], int]:
    """
    무엇을: 지금까지 기록된 매칭 이력(사용자 쌍 키 -> 매칭 횟수)을 반환한다.

    왜 복사본을 반환하는가: 호출자(server.py)가 반환된 dict를 자유롭게 matching.py에
    넘겨 쓰되, 그 dict를 직접 변형해도 저장소 내부 상태가 save_match_history() 호출 전에는
    바뀌지 않도록 하기 위함이다. "읽기"와 "쓰기"를 분리해두면, 나중에 실제 DB로 바뀌었을 때도
    같은 패턴(불러오기 -> 계산 -> 다시 저장)을 그대로 쓸 수 있다.
    """
    return dict(_match_history)


def save_match_history(history: Dict[Tuple[str, str], int]) -> None:
    """
    무엇을: 갱신된 매칭 이력 전체로 저장소 상태를 덮어쓴다.

    왜 함수로 감쌌는가: 지금은 단순히 모듈 전역 dict를 교체하는 것뿐이지만, 실제 DB로
    바꿀 때는 이 함수 내부만 "각 (user_a, user_b) 쌍을 upsert하는 쿼리 실행"으로 바꾸면
    되고, server.py의 호출 코드(store.save_match_history(match_history))는 전혀 손댈
    필요가 없다.
    """
    global _match_history
    _match_history = dict(history)


def get_restaurant_db() -> List[Dict[str, Any]]:
    """
    무엇을: 제휴 식당 목록을 반환한다. 지금은 ai_engine.demo.MOCK_RESTAURANT_DB를 초기값
    으로 복사해 두고 그걸 그대로 내려준다(나중에 실제 제휴 식당 DB 테이블 SELECT로 교체).

    왜 복사본을 반환하는가: restaurant.query_restaurants()는 조건에 맞는 각 후보 dict에
    "_distance_km" 같은 파생 키를 추가로 얹는다. 저장소가 들고 있는 원본 리스트를 그대로
    내주면 그 부작용이 저장소 상태 자체를 오염시키므로, 매 호출마다 깊은 복사본을 준다.
    """
    return copy.deepcopy(_restaurant_db)
