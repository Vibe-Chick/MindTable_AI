# MindTable_AI

```
backend/    편성 파이프라인 (Python, 의존성 없음)
            └ README.md     실행 방법
            └ FRONTEND.md   프론트엔드가 읽을 JSON 계약
frontend/   (예정)
```

## 브랜치 규칙

- `main` 은 직접 푸시하지 않는다. PR 로만 들어간다.
- 작업은 각자 브랜치에서: `backend/<작업>`, `frontend/<작업>`
- 서로 다른 폴더만 건드리면 충돌이 거의 없다.

## 백엔드 빠른 시작

```bash
cd backend
LLM_STUB=1 python3 run.py --mock 60 --rounds 3   # API 키 없이 전 구간
```

출력은 `backend/out/`. 프론트엔드는 이 JSON만 읽으면 된다 —
서버를 띄울 필요 없다. 자세한 건 `backend/FRONTEND.md`.
