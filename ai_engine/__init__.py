"""
ai_engine
==========
AI 기반 랜덤 식사 매칭 서비스의 핵심 로직 패키지.

원래 matching_engine.py 단일 파일이었던 것을, 백엔드 API가 필요한 부분만 골라 import하기
쉽도록 책임별로 분리했다. INTEGRATION_REPORT.md 기준으로 핵심 로직(llm_client/profile/
matching/feedback)은 김주환 브랜치를 이식해 교체했고, server/restaurant는 jaebin 원안을
유지했다 - 자세한 근거는 BRANCH_COMPARISON.md와 INTEGRATION_REPORT.md 참고.

모듈 구성:
    - llm_client : LLM 호출 추상화. 캐시+HTTP 상태코드 기반 재시도+게이트웨이 스키마 무시
                   감지(김주환 브랜치 이식) + 순정 Anthropic API 폴백(jaebin 유지)
    - schemas    : 온보딩/리뷰/매칭 결과에 쓰이는 데이터 구조 정의 (dataclass)
    - profile    : 온보딩 답변 -> 자기보고 성격 벡터(self_report_vector) 추출.
                   인용 그라운딩 검증(김주환 브랜치 이식)으로 근거 없는 판단을 걸러낸다.
    - matching   : 고정 태그 공간 유사도 + greedy/local search 편성 + 카드 생성(김주환
                   브랜치 이식), 매칭 이력 외부 저장 패턴은 jaebin 유지
    - restaurant : 확정된 그룹의 실제 식사 장소(제휴 식당) 확정 로직 (jaebin 유지)
    - feedback   : 식사 후 리뷰 기반 프로필 보정(EMA 감쇠 + 관심사 가중치 학습, 김주환
                   브랜치 이식) 파이프라인
    - server     : FastAPI 앱 (jaebin 유지, 내부 호출만 교체된 로직으로 재배선)
    - api_models / store : Pydantic 스키마 / 인메모리 저장소 (jaebin 유지)
    - demo       : 목데이터 기반 데모 실행부 (python -m ai_engine.demo)

각 모듈은 다른 모듈에 있는 전역 상태에 의존하지 않고, 필요한 값은 전부 인자로 주고받는다.
이 원칙 덕분에 백엔드 API 핸들러가 필요한 함수만 골라 import해서 그대로 쓸 수 있다.
"""
