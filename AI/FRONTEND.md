# 프론트엔드 계약

백엔드는 서버가 아니다. **`AI/out/` 의 정적 JSON 파일이 API다.**
파이프라인을 돌리면 이 파일들이 갱신된다. fetch 해서 그리면 된다.

샘플: **`AI/out/sample/`** (익명화됨). 이 구조 그대로 실데이터가 들어온다.

---

## AI/out/sample/groups.json

```jsonc
{
  "round": 2,
  "groups": [["u001","u008","u007","u005"], ...],   // user_id 배열
  "objective": { "random": -2.75, "greedy": -1.17, "local_search": -1.13 },
  "cards": [
    {
      "group_id": "g_r2_00",
      "members": [
        {
          "user_id": "u006",
          "display": "학생06",          // 표시명
          "school": "동국대",
          "major": "법학과",
          "college": "법과대학",
          "year": 4,
          "interests": ["토론", "역사"]  // 카드에 띄울 자유 키워드
        }
      ],
      "reason": "법학과 토론, 기계공학 요리, ...",   // 조합 이유 한 줄
      "icebreakers": ["질문1", "질문2", "질문3"],
      "icebreaker_targets": [["학생06","학생04",...], [...], [...]],
      "overlap": ["운동"],                          // 겹치는 태그
      "restaurant": { "name": "충무로 노포 분식", "category": "분식",
                      "price": "0.7만", "near": "동국대" }
    }
  ]
}
```

**그룹 카드 UI**: `reason` 을 헤더, `members` 를 칩, `icebreakers` 3개를
리스트, `restaurant` 를 하단 배지. `icebreaker_targets[i]` 는 그 질문에
답할 사람들 — 칩 하이라이트에 쓰면 좋다.

---

## AI/out/sample/profiles.json

```jsonc
[{
  "user_id": "u001",
  "display": "학생01",
  "school": "숭실대", "major": "소프트웨어학부",
  "college": "IT대학", "year": 2,

  "interests": ["홈서버","리눅스","재즈 LP"],     // 표시용 자유 키워드
  "tags": ["음악","기술"],                        // 유사도 계산용 고정 태그
  "interest_weights": { "음악": 1.41, "기술": 1.0 },  // 회차마다 학습됨

  "self_report": { "O":4,"C":3,"E":2,"A":3,"N":3 },   // 설문 결과, 불변
  "behavior":    { "O":0.22,"C":0,"E":0.45,"A":0,"N":0 }, // 리뷰로 누적된 보정

  "beta": 0.5,          // 개인별 다양성 선호 (리뷰 4번 문항으로 학습)
  "confidence": 0.75,
  "history": ["g_r1_00"],   // 참여한 그룹
  "met": ["u007","u004"]    // 같은 자리에 앉았던 사람
}]
```

### 핵심 시각화 — 이게 데모의 하이라이트다

```
현재값 = clamp(self_report + behavior, 1, 5)
```

`self_report`(설문)와 `현재값`(실제 자리 반영)의 **차이를 보여주는 화면**이
이 프로젝트의 주장 그 자체다.

```
학생01  외향성   설문 2  ─────●────→  현재 2.45
```

- 두 값을 한 축에 놓고 화살표로 잇는다
- 차이가 0.3 이상인 축만 강조
- `interest_weights` 도 같이: 설문엔 `기술`뿐이었는데 `음악`이 1.41로 올라감

---

## AI/out/metrics.json

발표 숫자의 **단일 출처**. 슬라이드에 손으로 옮겨 적지 말 것.

```jsonc
{
  "n_users": 60, "n_groups": 15,
  "objective": { "random": ..., "greedy": ..., "local_search": ... },
  "objective_gain_vs_random_pct": 50.1,
  "satisfaction_curve": [1.37, 1.34, 1.44],   // 회차별
  "topic_error_curve": [0.72, 0.55, 0.48, 0.42],
  "divergence_top": [                          // 설문과 가장 어긋난 사람들
    { "user_id":"u001", "axis":"E", "self_report":2,
      "effective":2.45, "delta":0.45 }
  ],
  "reliability": { "E": { "n":20, "pearson":0.8, "within_1":0.9 } },
  "interest_drift": [...],
  "cards_without_llm": 0
}
```

`AI/out/baseline_r1.json` 은 **랜덤 편성** 결과다. 알고리즘 편성과 나란히
띄우는 대조 화면이 데모의 핵심 장면이므로 반드시 구현할 것.

---

## 화면 우선순위 (시간 없으면 아래부터 버린다)

| 순위 | 화면 | 데이터 |
|---|---|---|
| 1 | **랜덤 vs 알고리즘 대조** (목적함수 값 병기) | `baseline_r1.json` + `groups.json` |
| 2 | 그룹 카드 | `groups.json` |
| 3 | 프로필 이동 (설문 → 현재) | `profiles.json` |
| 4 | 회차별 곡선 | `metrics.json` |
| 5 | 식당 표시 | `groups.json` |

---

## 주의

- **로그인·결제·모바일 대응 만들지 마라.** 배점 0이다.
- 값이 없을 수 있는 필드: `reason`(LLM 실패 시 null), `interest_vec`(빈 배열),
  `reliability`(수기 채점 전 null). 전부 방어적으로 처리할 것.
- `objective` 값은 **회차 간 비교 불가**다. 반복 매칭 페널티가 누적되어
  음수가 될 수 있다. 회차 비교에는 `satisfaction_curve` 를 쓸 것.
- 무대에서는 네트워크가 없다고 가정한다. JSON을 번들에 포함하거나
  로컬에서 읽어라.
