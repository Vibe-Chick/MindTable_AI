"""겸상(Venn) — 데이터 계약.

이 파일이 팀 전체의 인터페이스다. 여기 정의가 바뀌면 extract / match / review가
전부 영향을 받는다. 변경할 때는 반드시 팀에 공지할 것.

튜닝 상수도 전부 여기에만 둔다. 다른 파일에 숫자를 박지 말 것.
"""

# --- Big Five 축 ---
AXES = ("O", "C", "E", "A", "N")
AXIS_KR = {
    "O": "개방성", "C": "성실성", "E": "외향성",
    "A": "우호성", "N": "신경성",
}

SCORE_MIN, SCORE_MAX = 1, 5

# --- 편성 ---
GROUP_SIZE = 4          # To Do.md의 3~4인과 불일치. 4인으로 통일 결정 시 유지
ALPHA = 1.0             # 관심사 유사도 가중
GAMMA = 0.5             # 성격 보완도 가중
DELTA_REPEAT = 0.8      # 반복 매칭 페널티
EPSILON = 0.25          # 탐색 비율: 그룹당 1명은 유사도 하위 풀에서 충원

LOCAL_SEARCH_MAX_ITER = 500   # 최적해 불필요. 랜덤보다 낫다는 것만 보이면 됨

# --- 개인별 다양성 선호도 (리뷰 4번 문항으로 학습) ---
BETA_INIT = 0.5
BETA_STEP = 0.15
BETA_MIN, BETA_MAX = 0.1, 0.9

# --- 보정 루프 ---
# 아래 3개는 tune.py 그리드 서치 결과 (숨은 정답 시뮬레이션 기준).
# LR=0.9 가 개선율은 더 높았으나(+31%) 정확했던 프로필의 잡음 유입이 2배라
# 보수적인 조합을 택했다. 실데이터는 시뮬보다 잡음이 크므로 더 안전한 쪽.
LR = 0.5                # 델타 학습률
BEHAVIOR_CLIP = 1.5     # 누적 보정 상한. 없으면 3회차에 척도 밖으로 발산한다
BEHAVIOR_DECAY = 0.10   # 회차마다 기존 보정을 감쇠시킨다.
                        # 없으면 리뷰 잡음이 랜덤워크로 누적되어, 자기보고가
                        # 이미 정확했던 사람의 프로필까지 망가진다. (시뮬로 확인)
DELTA_DEADBAND = 0.60   # 이 미만의 신호는 0으로. 1회차 잡음에 반응하지 않는다
DIVERGENCE_THRESHOLD = 1.0   # 자기보고 vs 보정 괴리 하이라이트 기준

# --- 관심사 태그 (유사도 계산의 실제 공간) ---
# 자유 키워드만 쓰면 "클라이밍"과 "등산"이 남남이 된다. 임베딩이 없는
# 환경(국민대 게이트웨이에 임베딩 모델 없음)에서는 유사도가 0으로 붕괴한다.
# 그래서 고정 태그로 투영한 공간에서 유사도를 계산한다.
#   - tags     : 유사도 계산용 (겹친다)
#   - interests: 카드 표시용 자유 키워드 (구체적이다)
TAGS = ("운동", "여행", "요리", "음악", "영상", "독서",
        "게임", "기술", "학술", "창작", "봉사", "재테크")

# --- 관심사 학습 (리뷰 R1/R2에서) ---
# 설문에 '쓴' 관심사와 식사 자리에서 '실제로 통한' 주제는 다르다.
# 후자를 반영해야 유사도 항(가중치 ALPHA=1.0, 목적함수 최대항)이 움직인다.
INTEREST_UP = 0.45       # 대화가 터진 주제 가중 상승
INTEREST_DOWN = 0.30     # 죽은 주제 가중 하락
INTEREST_DECAY = 0.08    # 회차마다 1쪽으로 수축 (성격 보정의 감쇠와 같은 역할)
INTEREST_NEW = 0.25      # 설문에 없던 태그가 처음 등장할 때 초기 가중.
                         # 설문 태그가 1.0 이므로 한 번의 식사가 설문과
                         # 맞먹지 않게 낮게 잡는다. 반복 등장하면 올라간다.
INTEREST_MIN, INTEREST_MAX = 0.0, 2.0
INTEREST_DROP = 0.05     # 이 미만이면 목록에서 제거


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def new_profile(user_id, school, major, college, year,
                self_report=None, interests=None, tags=None):
    """빈 프로필 생성. behavior는 항상 0에서 시작한다."""
    return {
        "user_id": user_id,
        "school": school,
        "major": major,
        "college": college,
        "year": year,
        # 설문 1회차 결과. 이후 절대 수정하지 않는다 (발표 시각화의 기준선)
        "self_report": self_report or {a: 3 for a in AXES},
        # 리뷰로 누적되는 보정값
        "behavior": {a: 0.0 for a in AXES},
        "interests": interests or [],          # 표시용 자유 키워드
        # 태그별 가중. 설문 태그는 1.0에서 시작하고 리뷰가 이 값을 움직인다.
        # 유사도는 전적으로 이 벡터로 계산한다.
        "interest_weights": {t: 1.0 for t in (tags or [])},
        "tags": list(tags or []),
        "interest_vec": [],
        "beta": BETA_INIT,
        "confidence": 0.0,
        "history": [],          # 참여한 group_id
        "met": [],              # 같은 자리에 앉았던 user_id (반복 페널티용)
    }


def effective(p):
    """실제 매칭에 쓰이는 벡터 = 자기보고 + 행동 보정."""
    return {
        a: clamp(p["self_report"][a] + p["behavior"][a], SCORE_MIN, SCORE_MAX)
        for a in AXES
    }


def divergence(p):
    """자기보고와 보정 후의 축별 괴리. 발표 5번 장면의 재료."""
    eff = effective(p)
    return {a: eff[a] - p["self_report"][a] for a in AXES}


def max_divergence(p):
    d = divergence(p)
    a = max(d, key=lambda k: abs(d[k]))
    return a, d[a]


def validate_profile(p):
    """스키마 위반을 조용히 넘기지 않는다. 실패는 즉시 터뜨린다."""
    errs = []
    for f in ("user_id", "school", "major", "college", "year",
              "self_report", "behavior", "interests", "beta", "confidence"):
        if f not in p:
            errs.append(f"missing field: {f}")
    if errs:
        return errs
    for a in AXES:
        v = p["self_report"].get(a)
        if not isinstance(v, int) or not (SCORE_MIN <= v <= SCORE_MAX):
            errs.append(f"self_report.{a} invalid: {v!r}")
        b = p["behavior"].get(a)
        if not isinstance(b, (int, float)) or abs(b) > BEHAVIOR_CLIP + 1e-9:
            errs.append(f"behavior.{a} out of clip: {b!r}")
    if not (BETA_MIN - 1e-9 <= p["beta"] <= BETA_MAX + 1e-9):
        errs.append(f"beta out of range: {p['beta']!r}")
    if not (0.0 <= p["confidence"] <= 1.0):
        errs.append(f"confidence out of range: {p['confidence']!r}")
    return errs


# --- LLM 구조화 출력 스키마 ---------------------------------------------

_SC = {"type": "integer", "minimum": 1, "maximum": 5}
_Q = {"type": "string"}

# 평탄한 스키마. 중첩 object 는 haiku 급 모델에서 절반쯤 누락된다(실측).
# 축마다 점수와 '원문 인용'을 쌍으로 받고, extract.py 가 인용이 실제 답변에
# 들어있는지 대조한다. 인용이 없거나 원문에 없으면 점수를 3으로 강제한다.
EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "openness": _SC,          "openness_quote": _Q,
        "conscientiousness": _SC, "conscientiousness_quote": _Q,
        "extraversion": _SC,      "extraversion_quote": _Q,
        "agreeableness": _SC,     "agreeableness_quote": _Q,
        "neuroticism": _SC,       "neuroticism_quote": _Q,
        "interests": {"type": "array", "items": {"type": "string"},
                      "minItems": 1, "maxItems": 4},
        # interests 와 같은 길이. 각 키워드의 근거 인용.
        "interest_quotes": {"type": "array", "items": {"type": "string"},
                            "minItems": 1, "maxItems": 4},
        # 고정 태그로의 투영. 유사도는 이걸로 계산한다.
        "tags": {"type": "array",
                 "items": {"type": "string", "enum": list(TAGS)},
                 "minItems": 1, "maxItems": 3},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["openness", "openness_quote",
                 "conscientiousness", "conscientiousness_quote",
                 "extraversion", "extraversion_quote",
                 "agreeableness", "agreeableness_quote",
                 "neuroticism", "neuroticism_quote",
                 "interests", "interest_quotes", "tags", "confidence"],
}

_DELTA = {"type": "number", "enum": [-1, -0.5, 0, 0.5, 1]}
_TOPICS = {"type": "array",
           "items": {"type": "string", "enum": list(TAGS)},
           "minItems": 0, "maxItems": 3}

DELTA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "openness": _DELTA, "conscientiousness": _DELTA,
        "extraversion": _DELTA, "agreeableness": _DELTA,
        "neuroticism": _DELTA,
        # 실제로 대화가 터진 주제 / 죽은 주제.
        # 이게 유사도 행렬을 갱신한다 — 보정이 지배적 항에 닿는 유일한 경로.
        "worked_topics": _TOPICS,
        "dead_topics":   _TOPICS,
        "evidence":   {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["openness", "conscientiousness", "extraversion",
                 "agreeableness", "neuroticism", "worked_topics",
                 "dead_topics", "evidence", "confidence"],
}

LONG_TO_AXIS = {
    "openness": "O", "conscientiousness": "C", "extraversion": "E",
    "agreeableness": "A", "neuroticism": "N",
}


CARD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reason": {"type": "string"},
        # 각 질문마다 '누가 답할 수 있는지'를 함께 받는다.
        # 프롬프트로 세 번 고쳐도 1인 전용 질문이 새어나와서, 코드로 검증한다.
        "icebreakers": {
            "type": "array",
            "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "targets": {"type": "array",
                                "items": {"type": "string"},
                                "minItems": 1, "maxItems": 6},
                },
                "required": ["question", "targets"],
            },
        },
    },
    "required": ["reason", "icebreakers"],
}
