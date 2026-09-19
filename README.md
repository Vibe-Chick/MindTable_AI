# 겸상 (Venn) — 백엔드

의존성 없음. Python 3.8+ stdlib만. `pip install` 불필요.

## 환경

```bash
export LLM_PROVIDER=openai
export LLM_BASE_URL=https://ai.cs.kookmin.ac.kr/v1
export LLM_MODEL=claude-haiku-4-5
export LLM_API_KEY=<키>
```

`.env`에 넣고 `source .env`. gitignore 되어 있다.

**게이트웨이 주의**: 국민대 게이트웨이는 `response_format: json_schema` 와
`json_object` 를 **에러 없이 무시한다**. `tools` 방식만 동작한다
(`LLM_STRUCTURED=tools`, 기본값). `probe.py` 로 확인 가능.
임베딩 모델은 없다 → 고정 태그 공간으로 유사도를 계산한다.

## 실행 순서

```bash
python3 src/extract.py data/responses.csv     # 설문 → 프로필  (N콜)
python3 run_real.py 1                         # 편성 + 카드    (그룹수 콜)
python3 make_reviews.py 1                     # 리뷰 템플릿 생성
                                              # ← 사람이 채운다
python3 round_close.py 1                      # 리뷰 → 프로필 갱신 (N콜)
python3 run_real.py 2                         # 2회차 편성
python3 compare_rounds.py 1 2                 # 회차 간 변화
```

개발·데모용 (API 0콜):

```bash
LLM_STUB=1 python3 run.py --mock 60 --rounds 3   # 전 구간
python3 src/match.py data/profiles_mock.json     # 최적화기 벤치
python3 src/verify.py 4                          # 8개 시드 루프 검증
python3 src/tune.py                              # 파라미터 그리드 서치
```

## 구조

```
src/schema.py    데이터 계약 + 튜닝 상수 (숫자는 여기에만)
src/llm.py       LLM 호출: 캐시·재시도·동시성·스텁·스키마 무시 감지
src/prompts.py   프롬프트 (분리)
src/extract.py   ★ LLM #1  설문 → Big Five + 태그
src/embed.py     태그 가중 벡터 → 유사도 행렬
src/match.py     편성 최적화기 (LLM 아님)
src/cards.py     ★ LLM #2  조합 이유 + 아이스브레이커
src/review.py    ★ LLM #3  리뷰 → 프로필 델타
src/metrics.py   발표 숫자 집계
src/simulate.py  회차 시뮬레이션 + 보정 ON/OFF 대조
src/mock.py      합성 프로필 (숨은 정답 포함)
```

## 모델 출력을 믿지 않는 지점 (검증 5종)

프롬프트로 지시했지만 모델이 지키지 않아, 코드에서 강제한 것들.
전부 실호출에서 실제로 겪고 나서 추가했다.

| 단계 | 모델이 안 지킨 것 | 방어 |
|---|---|---|
| 추출 | 원문 그대로 인용 | 인용문을 답변 원문과 문자열 대조 |
| 추출 | 근거 없으면 3점 | 인용 없으면 코드가 3점으로 덮어씀 |
| 추출 | 지어내지 마라 | 근거 없는 관심사 제거 |
| 카드 | 전원이 답할 질문 | `targets` 검증 + 1회 재작성 |
| 카드 | 이 그룹 전용 질문 | 활동 키워드 포함 여부 검사 |

추가로 `llm.py` 가 응답의 required 키를 검사해 **게이트웨이의 스키마 무시**를
런타임에 잡는다.

## 규칙

- 튜닝 상수는 `schema.py` 에만. 다른 파일에 숫자를 박지 말 것.
- 모든 LLM 결과는 `cache/` 에 파일로 남는다. 재실행은 공짜다.
  **프롬프트를 고치면 캐시 키가 바뀌어 전량 재호출된다.**
- **무대에서는 API를 호출하지 않는다.** `LLM_CACHE_ONLY=1` 로 확인할 것.
- `WORKERS=4` 고정. 신규 키는 동시 요청이 많으면 429.
- `interests`(표시용 자유 키워드)와 `interest_weights`(태그 가중, 유사도용)는
  다른 것이다. 학습은 태그만 건드린다.

## mock 기준 수치

```
최적화기        random 대비 +50%
보정 ON vs OFF  중앙값 +10.95%  (8개 시드 전부 양수)
관심사 오차     -47.5%
```

※ 실데이터 수치는 60건 수집 후 다시 뽑아야 한다.

## 다음

`data/responses.csv` (실제 응답 40~60건). 그 외 블로커 없음.
설문 원고는 `../설문지.md`.
