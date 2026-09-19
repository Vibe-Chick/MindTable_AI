# jaebin ↔ 김주환 통합 보고서 (integration 브랜치)

팀 결정: **서빙 방식은 jaebin의 FastAPI 서버를 유지하고, 내부 파이프라인 로직은
김주환 브랜치 것으로 교체한다** (BRANCH_COMPARISON.md 최종 권장안 기준). 이
문서는 그 통합을 실제로 수행한 결과를 정리한다.

- 작업 브랜치: `integration` (jaebin에서 분기, jaebin/김주환 원본 브랜치는 건드리지 않음)
- 변경 규모: 14개 파일, +1,514 / -1,020줄 (`json_utils.py` 삭제 포함)
- 0~6번(필수) 전부 완료, 8번(검증) 전부 완료, 7번(mock 데이터 대개편)은 시간 관계상
  스킵 — 단, mock 데이터에 새 로직이 요구하는 최소 필드(`tags`/`interest_weights`/
  `diversity_beta`)는 3단계 작업 중 추가해 데모가 정상 동작한다.

---

## 1. 무엇을 어디서 가져왔는지

| 파일 | 이번 통합에서 한 일 |
|---|---|
| `ai_engine/llm_client.py` | 김주환 `AI/src/llm.py` 전체 이식(캐시/HTTP상태코드 재시도/게이트웨이 스키마 무시 감지/STUB/CACHE_ONLY) + jaebin의 순정 Anthropic API를 **폴백 경로**로 유지 + `ANTHROPIC_*`↔`LLM_*` 환경변수 별칭 처리 |
| `ai_engine/profile.py` | 김주환 `AI/src/extract.py`+`prompts.py`의 **인용 그라운딩 검증**을 이식, jaebin의 q1~q4_choice 설문 형식/필드명(영문 풀네임)은 그대로 유지 |
| `ai_engine/matching.py` | 김주환 `AI/src/embed.py`(고정 태그 공간 유사도) + `AI/src/match.py`(축별 차등 성격 보완도, greedy+local search) + `AI/src/cards.py`(타겟 커버리지 검증) 이식. `run_matching_pipeline` 시그니처/반환 형태와 매칭 이력 외부 저장 패턴(`store.py`)은 jaebin 그대로 유지 |
| `ai_engine/feedback.py` | 김주환 `AI/src/review.py` 전체 이식 (EMA 감쇠, 관심사 가중치 학습). `run_feedback_pipeline`의 시그니처/반환 형태는 jaebin 그대로 유지 |
| `ai_engine/restaurant.py` | jaebin 원안(haversine+예산 교집합+조건 완화) 그대로 유지, 응답에 김주환/FRONTEND.md 계약(`name/category/price/near`) 필드를 추가하는 어댑터만 얹음 |
| `ai_engine/schemas.py` | `UserProfile`에 `tags`/`interest_weights`/`confidence` 필드 추가, `MatchGroup`에 `icebreaker_targets`/`overlap` 추가, `TAGS`(고정 12종) 상수 추가. **`behavior_corrected_vector`의 의미를 "절대값 복사"에서 "0에서 시작하는 오프셋"으로 변경**(아래 §3 버그 참고) |
| `ai_engine/server.py`, `api_models.py` | 그대로 유지, 내부 호출만 새 로직에 맞게 재배선(`_candidate_to_matching_dict`가 self_report+behavior 오프셋을 더해 effective 벡터를 계산하도록 수정) + `/admin/cache-only` 엔드포인트 신설 |
| `ai_engine/demo.py` | mock 유저에 `tags`/`interest_weights`/`diversity_beta` 추가, 식당 DB에 `category` 추가, 리뷰 데모의 `r4_choice`를 새 enum 값으로 수정, 케이스4(임계값 검증)를 EMA 방식으로 재작성 |
| `ai_engine/json_utils.py` | **삭제**. `llm_client.py`가 JSON 추출 로직을 자체 내장(김주환 방식)하면서 더 이상 쓰이지 않는 죽은 코드가 됨 |
| `.env.example`, `requirements.txt`, `.gitignore` | 새 환경변수 체계 문서화, `anthropic`/`openai`/`voyageai` SDK 의존성 제거(김주환처럼 stdlib만으로 LLM 호출), `cache/` 디렉터리 gitignore 추가 |

---

## 2. 최종 채택 정책 (숫자 근거 포함)

| 정책 | 채택값 | 근거 |
|---|---|---|
| 반복 매칭 페널티 | 이진 방식, 쌍당 **-0.8** (만난 적 있으면 무조건, 횟수 무관) | 김주환 브랜치가 숨은 정답 시뮬레이션(`tune.py` 그리드서치)으로 검증한 값. jaebin 원안(횟수비례 -0.1×n, 상한 -0.3)보다 공격적이지만 실측 근거가 있는 쪽을 우선 |
| 보정 학습률(LR) | **0.5** | 동일 그리드서치. LR=0.9는 개선율이 더 높았으나(+31%) 이미 정확했던 프로필의 잡음 유입이 2배라 기각됨(김주환 브랜치 `schema.py` 주석 근거) |
| 보정 감쇠(BEHAVIOR_DECAY) | **0.10** (회차마다 기존 보정치의 10%를 자연 감쇠) | 위와 동일 그리드서치. 감쇠 없으면 3회차 안에 척도 밖으로 발산함이 시뮬레이션으로 확인됨 |
| 보정 누적 상한(BEHAVIOR_CLIP) | **±1.5** | 동일 |
| 다양성 β 조정폭 | ±0.15, 범위 0.1~0.9 | jaebin/김주환 두 브랜치가 **독립적으로 동일한 값**에 도달함(BRANCH_COMPARISON.md에서 이미 지적된 수렴적 설계) - 그대로 유지 |
| 카드 생성 병렬 처리 | `ThreadPoolExecutor(max_workers=8)` | **아래 §4 참고 - 실측 미완료, 잠정값** |
| 배치 호출(프로필 추출 등) 동시성 | `LLM_WORKERS=4` (환경변수로 조정 가능) | 김주환 브랜치가 "신규 키는 동시 요청 많으면 429"라고 명시한 값을 기본값으로 채택 |

---

## 3. 통합 과정에서 발견하고 고친 버그

이식 도중 발견한 실제 버그이며, 단순 포팅이 아니라 검증을 거쳐 수정했다.

**`behavior_corrected_vector`의 의미 불일치.** jaebin 원안은 이 필드를
self_report_vector의 **복사본**(절대 점수, 예: openness=4)으로 초기화하고
누적+클립 방식으로 갱신했다. 김주환의 EMA 공식(`behavior*(1-0.1) + 0.5*delta`,
클립 범위 ±1.5)을 그대로 이식하면서, 이 필드가 김주환 쪽에서는 **0에서 시작하는
오프셋**이라는 점을 처음에 반영하지 않았다. 그 결과 절대값(예: 4)에 오프셋 전용
클립(±1.5)이 즉시 적용되어 `behavior_corrected_vector`가 1.5로 truncate되고,
`compute_discrepancy`가 `|1.5 - 4| = 2.5`라는 잘못된 괴리값을 내는 버그가
데모 실행 중 실제로 재현됐다(케이스4에서 `2.50`이 정확히 이 버그의 산출값이었다).

고친 내용:
1. `UserProfile.from_extracted_profile`: `behavior_corrected_vector`를 축마다
   `0.0`으로 초기화(절대값 복사 아님).
2. `feedback.compute_discrepancy`: `behavior` 오프셋을 그대로 괴리로 쓰지 않고,
   `effective = clamp(self_report + behavior, 1, 5)`를 다시 계산한 뒤
   `|effective - self_report|`로 정의(척도 경계에서 클램프가 괴리를 깎아먹는
   경우를 올바르게 반영 - 김주환 `divergence()`와 동일 공식).
3. `server._candidate_to_matching_dict`: `behavior_corrected_vector`를 곧바로
   매칭 벡터로 쓰던 부분을 `self_report + behavior` 합산 후 클램프로 수정
   (안 고쳤으면 리뷰 피드백을 받은 유저가 매칭 시 거의 0에 가까운 성격
   점수로 계산될 뻔했다).

수정 후 데모 재실행으로 케이스4가 정확히 `1.00`(기대값과 일치)을 내는 것을
확인했다(§5 참고).

---

## 4. 카드 생성 병렬화(max_workers=8) 실측 — **미완료, 확인 필요**

이 세션에는 **유효한 LLM API 키가 없었다**(이전 세션에서 보안상 이미 폐기된
키만 있었고, 재발급된 키를 채팅에 붙여넣지 않도록 안내했기 때문에 이번에도
실제 게이트웨이 호출은 하지 못했다). 따라서:

- `max_workers=8`로 실제 게이트웨이에 대량 동시 요청을 보내 429가 뜨는지 확인하는
  작업은 **수행하지 못했다.**
- 대신 `LLM_STUB=1`과 규칙 기반 폴백 경로로 로직 자체(병렬 실행 시 결과 순서 보존,
  개별 그룹 실패 격리)는 검증했다.
- `LLM_WORKERS=4`(배치 호출)와 `MAX_LLM_WORKERS=8`(카드 생성)이 서로 다른 상수로
  분리되어 있으므로, 실측 후 어느 한쪽만 낮추는 것도 가능하다.

**팀에게 요청**: 실제 게이트웨이 키로 8명 이상(그룹 2개↑) mock 데이터를 돌려서
429가 뜨는지 확인해달라. 뜨면 `ai_engine/matching.py`의 `MAX_LLM_WORKERS` 상수를
4로 낮추면 된다(한 줄 수정).

---

## 5. 검증 결과 (8단계, 전부 수행)

1. **`python -m ai_engine.demo` 전체 파이프라인**: 정상 종료(exit 0). 매칭
   0.005~0.024초(폴백 경로), 식당 확정 0단계 완화 성공, 리뷰 피드백 4개 케이스
   전부 기대값과 일치(특히 케이스4 discrepancy=1.00, `profile_shift_event=True`).
2. **FastAPI 서버 4개 엔드포인트 + 신규 2개**: `/health`(+캐시 상태 노출),
   `/admin/cache-only`(런타임 토글), `/profile/extract`, `/match/run`(8명→그룹2개),
   `/restaurant/resolve`(카테고리/가격/near 필드 포함 정상 반환),
   `/review/submit`(diversity_beta 0.5→0.65 반영 확인) — **전부 200, 실제 호출로
   확인**.
3. **식당 매칭 성공률 재확인**: 8명 중 4명 조합 70가지 **전부 완화 0단계**로
   성공(통합 이전과 동일한 결과 - 좌표/DB 로직을 건드리지 않았으므로 회귀 없음
   확인).
4. **수렴 검증 (simulate.py/verify.py 축소판)**: (a) 자기보고가 틀린 축(2점,
   실제 5)에 일관된 신호를 5회 반영하자 오차가 `3.0 → 1.624`로 단조 감소함을
   확인(EMA가 실제로 수렴 방향으로 작동). (b) 이미 정확한 프로필에 약한
   confidence(0.3)의 무작위 신호를 5회 넣어도 최대 drift가 0.308에 그쳐
   잡음에 강함을 확인.

모든 검증은 실행 후 즉시 확인했고, 스크립트는 검증 목적의 일회성 파일이라
작업 완료 후 삭제했다(레포에 남기지 않음).

---

## 6. 팀원들이 알아야 할 변경사항

### 환경변수
- 기존 `.env`(ANTHROPIC_API_KEY 등)는 **그대로 써도 된다** - `llm_client.py`가
  별칭으로 자동 인식한다.
- 새 팀원은 `LLM_PROVIDER`/`LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL` 체계 사용을
  권장(`.env.example` 참고).
- `LLM_STUB=1`: API 없이 개발(가짜 응답, 캐시에 안 남음).
- `LLM_CACHE_ONLY=1`: 캐시에 없으면 에러(무대 시연 직전 필수) - 서버 재시작 없이
  `POST /admin/cache-only {"enabled": true}`로도 전환 가능.
- **의존성 감소**: `pip install`에서 `anthropic`/`openai`/`voyageai`가 빠졌다
  (`requirements.txt` 갱신됨). LLM 호출이 stdlib만으로 동작하므로 새로 `pip
  install -r requirements.txt`를 다시 실행해도 되고, 안 해도 된다(빠진 패키지는
  더는 import되지 않는다).

### 데이터 스키마
- `UserProfile`에 `tags`(고정 12종 분류), `interest_weights`(태그별 학습 가중치),
  `confidence`가 추가됐다. `/match/run`에 후보를 보낼 때 `tags`/`interest_weights`를
  안 보내도 동작은 하지만(빈 리스트로 폴백), **관심사 유사도가 전부 0으로
  계산된다** - 프론트/다른 백엔드 연동 시 이 필드를 꼭 채워서 보낼 것.
- `behavior_corrected_vector`는 이제 **절대 점수가 아니라 오프셋**이다(§3 참고).
  이 필드를 직접 읽어 "이 사람의 현재 성격 점수"로 쓰던 코드가 있다면
  `self_report_vector[axis] + behavior_corrected_vector[axis]`(1~5 클램프)로
  고쳐야 한다.
- `/match/run` 응답과 `/restaurant/resolve` 응답에 필드가 늘었다(`icebreaker_targets`,
  `overlap`, 식당의 `category`/`price`/`near`). 기존 필드는 그대로 남아있으므로
  하위 호환은 깨지지 않는다.
- 리뷰 강제선택(`r4_choice`) 답변은 이제 **정확히 세 문자열 중 하나**여야
  `diversity_beta`가 움직인다: `"더 비슷"` / `"지금 정도"` / `"더 달라도 됨"`
  (`ai_engine/feedback.py`의 `Q4_MORE_SIMILAR`/`Q4_SAME`/`Q4_MORE_DIFFERENT`).
  프론트는 자유 텍스트가 아니라 이 3개 버튼만 노출해야 한다.
- 리뷰 자유서술 질문(r1/r2/r3)의 **의도된 내용이 바뀌었다**: 기존엔 "식사가
  어땠는지" 일반 소감이었지만, 이제는 각각 "대화가 잘 풀린 순간+무슨 얘기였는지"
  / "말이 끊긴 주제" / "이끄는 쪽이었는지 듣는 쪽이었는지"로 더 구체적이다
  (관심사 학습 루프가 이 구체성을 필요로 한다). 설문 문구를 담당하는 팀원에게
  전달 필요.

### 코드
- `matching.py`의 `personality_similarity`/`diversity_score`/`get_embedding`/
  `pair_score`/`form_groups`(구현) 함수는 사라지고 `personality_complement`/
  `diversity`/`similarity_matrix`/`score_group`/`form_groups`(새 구현, 같은
  이름 유지)로 교체됐다. `run_matching_pipeline`을 통해서만 쓰던 코드는 영향
  없음.
- `json_utils.py`가 삭제됐다. 혹시 어딘가에서 `from ai_engine.json_utils import
  ...`를 하고 있다면(현재 레포 안에는 없음을 확인함) 고쳐야 한다.

---

## 7. 남은 TODO

- [ ] **최우선**: 실제 게이트웨이 키로 `MAX_LLM_WORKERS=8` 429 여부 실측 (§4)
- [ ] 7단계(mock 데이터 대개편 - 숨은 정답 내장, 전공-관심사 상관관계)는 스킵됨.
      데모에는 지장 없지만, 발표에서 "보정 루프가 실제로 수렴한다"는 정량적
      그래프가 필요하면 김주환 브랜치의 `mock.py`/`simulate.py`/`verify.py`를
      별도로 이식해야 한다.
- [ ] `restaurant.py`의 `_to_card_shape`가 `near`를 "그룹 내 최다 소속 학교"로
      근사하는데, 실제로는 "식당이 어느 학교에 가까운가"가 더 정확한 의미다.
      필요하면 MOCK_RESTAURANT_DB 각 항목에 `near` 필드를 직접 박아두는 게
      더 정확하다(현재는 파생값).
- [ ] `LLM_STRUCTURED`가 게이트웨이 종류에 따라 자동 전환되지 않는다(수동
      환경변수). 여러 게이트웨이를 혼용할 계획이면 `probe.py`(김주환 브랜치,
      아직 이식 안 함)를 가져와 자동 감지를 추가하는 것도 고려할 만하다.
- [ ] 홈 디렉터리 전체가 git 저장소로 잡혀 있는 사전 문제(이전 세션에서 발견,
      `mind_table`과 무관)는 여전히 미해결 - 별도로 처리 필요.

---

## 8. 브랜치 상태

- `integration` 브랜치는 origin에 push됨(아래 커밋 참고).
- **`main`에는 병합하지 않았다** - 사용자 최종 승인 후 진행 예정.
- `jaebin`, `김주환` 원본 브랜치는 이번 작업에서 전혀 수정하지 않았다(참조용
  worktree `../ref-kimjuhwan`만 읽기 목적으로 사용, 작업 종료 후 제거).
