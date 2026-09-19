# 겸상 (Venn) — 백엔드

의존성 없음. Python 3.8+ stdlib만 사용. `pip install` 불필요.

## 구조

```
src/schema.py   데이터 계약 + 튜닝 상수 (여기만 고친다)
src/llm.py      LLM 호출: 캐시 + 재시도 + 동시성 제한
src/mock.py     합성 프로필 (실제 응답 없이 개발 시작용)
src/embed.py    관심사 임베딩 → 유사도 행렬
src/match.py    편성 최적화기 (LLM 아님)
```

## 환경변수

```bash
export LLM_PROVIDER=openai        # openai | anthropic | gemini
export LLM_MODEL=<모델ID>          # 벤더 페이지에서 확인 후 설정
export LLM_API_KEY=<키>
export LLM_CACHE_ONLY=1           # 무대 시연 때 반드시 켠다
```

## 지금 동작하는 것

```bash
python3 src/mock.py 60 > data/profiles_mock.json
python3 src/match.py data/profiles_mock.json
```

API 키 없이 최적화기 전체를 벤치마크할 수 있다.

## 규칙

- 튜닝 상수는 `schema.py`에만 둔다. 다른 파일에 숫자를 박지 말 것.
- 모든 LLM 결과는 `cache/`에 파일로 남는다. 재실행은 공짜다.
- **무대에서는 API를 호출하지 않는다.** `LLM_CACHE_ONLY=1`로 확인할 것.
- `WORKERS=4` 고정. 신규 키는 동시 요청이 많으면 429.

## 아직 없는 것

`extract.py` (Stage 2) · `cards.py` (Stage 6) · `review.py` (Stage 7) · `metrics.py` (Stage 3) · `run.py`
