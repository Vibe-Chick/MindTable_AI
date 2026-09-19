"""
ai_engine
==========
AI 기반 랜덤 식사 매칭 서비스의 핵심 로직 패키지.

원래 matching_engine.py 단일 파일이었던 것을, 백엔드 API가 필요한 부분만 골라 import하기
쉽도록 책임별로 분리했다.

모듈 구성:
    - llm_client : Claude/OpenAI 호환 게이트웨이 호출 추상화 (call_llm, get_chat_backend)
    - json_utils : LLM 응답에서 JSON 객체/배열만 잘라내는 파싱 방어 유틸리티
    - schemas    : 온보딩/리뷰/매칭 결과에 쓰이는 데이터 구조 정의 (dataclass)
    - profile    : 온보딩 답변 -> 자기보고 성격 벡터(self_report_vector) 추출
    - matching   : 유사도/다양성 계산, 그룹 구성, 매칭 이유/아이스브레이커 생성, 매칭 이력 페널티
    - restaurant : 확정된 그룹의 실제 식사 장소(제휴 식당) 확정 로직
    - feedback   : 식사 후 리뷰 기반 프로필 보정(행동 벡터 업데이트) 파이프라인
    - demo       : 목데이터 기반 데모 실행부 (python -m ai_engine.demo)

각 모듈은 다른 모듈에 있는 전역 상태에 의존하지 않고, 필요한 값은 전부 인자로 주고받는다.
이 원칙 덕분에 백엔드 API 핸들러가 필요한 함수만 골라 import해서 그대로 쓸 수 있다.
"""
