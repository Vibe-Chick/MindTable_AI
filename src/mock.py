"""합성 프로필 생성기.

목적: C(최적화기 담당)가 실제 설문 응답을 기다리지 않고 바로 작업을 시작할 수 있게 한다.
관심사를 테마별로 뭉치게 생성하므로 유사도 행렬에 실제 구조가 생긴다.
(완전 랜덤으로 뽑으면 유사도가 균일 노이즈가 되어 최적화기 검증이 불가능하다.)

    python3 src/mock.py 60 > data/profiles_mock.json
"""
import json
import random
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from schema import (AXES, BETA_INIT, BETA_MAX, BETA_MIN, TAGS,  # noqa: E402
                    new_profile, validate_profile)

SCHOOLS = {
    "숭실대": [("IT대학", "소프트웨어학부"), ("IT대학", "전자정보공학부"),
             ("경영대학", "경영학부"), ("인문대학", "영어영문학과")],
    "중앙대": [("소프트웨어대학", "소프트웨어학부"), ("경영경제대학", "경영학부"),
             ("예술대학", "공연영상창작학부")],
    "한양대": [("공과대학", "기계공학부"), ("공과대학", "컴퓨터소프트웨어학부"),
             ("사회과학대학", "미디어커뮤니케이션학과")],
    "건국대": [("공과대학", "화학공학부"), ("문과대학", "철학과"),
             ("수의과대학", "수의예과")],
    "서울여대": [("과학기술융합대학", "데이터사이언스학과"), ("인문대학", "사학과")],
    "동국대": [("이과대학", "수학과"), ("법과대학", "법학과"),
             ("예술대학", "미술학부")],
}

# 테마별 관심사 풀 — 같은 테마끼리 유사도가 높게 나와야 한다
THEMES = {
    "운동":   ["클라이밍", "러닝", "헬스", "수영", "주짓수", "등산"],
    "영상":   ["일본영화", "다큐멘터리", "독립영화", "애니메이션", "영화제"],
    "기술":   ["홈서버", "자작PC", "라즈베리파이", "리눅스", "보안", "오픈소스"],
    "음악":   ["재즈", "밴드", "작곡", "LP수집", "공연관람"],
    "요리":   ["베이킹", "홈카페", "위스키", "커피", "맛집탐방"],
    "여행":   ["배낭여행", "국내여행", "캠핑", "자전거여행", "사진"],
    "독서":   ["소설", "철학", "역사", "북클럽", "에세이"],
    "게임":   ["보드게임", "인디게임", "FPS", "롤", "방탈출"],
}
THEME_NAMES = list(THEMES)

FIRST = ["학생"]   # 표시명은 mock 생성 시 번호를 붙인다

# 전공 → 관심사 성향. 실제 데이터에는 이 상관이 존재한다.
# 이걸 넣지 않으면 유사도와 다양성이 서로 부딪히지 않아서, 최적화기가
# 쉬운 문제를 푸는 셈이 되고 벤치마크 숫자가 과대평가된다.
MAJOR_AFFINITY = [
    (("소프트웨어", "컴퓨터", "전자", "데이터"), ["기술", "게임"]),
    (("경영",),                                 ["여행", "요리"]),
    (("미술", "공연영상", "예술"),               ["영상", "음악"]),
    (("철학", "사학", "영어영문", "법학"),        ["독서", "영상"]),
    (("기계", "화학"),                          ["운동", "기술"]),
    (("수의", "수학", "미디어"),                 ["운동", "독서"]),
]


def _affinity(major):
    for keys, themes in MAJOR_AFFINITY:
        if any(k in major for k in keys):
            return themes
    return THEME_NAMES


def gen(n=60, seed=42):
    rng = random.Random(seed)
    profiles = []
    for i in range(n):
        school = rng.choice(list(SCHOOLS))
        college, major = rng.choice(SCHOOLS[school])

        # 주 테마는 전공 성향에서 65% 확률로, 나머지는 무작위
        aff = _affinity(major)
        main = rng.choice(aff) if rng.random() < 0.65 else rng.choice(THEME_NAMES)
        sub = rng.choice([t for t in THEME_NAMES if t != main])
        interests = rng.sample(THEMES[main], 2) + [rng.choice(THEMES[sub])]

        # 성격: 정규분포 비슷하게, 1~5 정수
        self_report = {}
        for a in AXES:
            v = round(rng.gauss(3.0, 0.9))
            self_report[a] = max(1, min(5, v))

        tags = [main] + ([sub] if rng.random() < 0.5 else [])
        p = new_profile(
            user_id="u%03d" % (i + 1),
            school=school, major=major, college=college,
            year=rng.choice([1, 1, 2, 2, 3, 4]),
            self_report=self_report,
            interests=interests,
            tags=[t for t in tags if t in TAGS],
        )
        p["name"] = "학생%02d" % (i + 1)        # 데모 화면용 표시명
        p["theme"] = main                      # 검증용. 실제 파이프라인엔 없음

        # --- 숨은 정답 (시뮬레이션 전용) ------------------------------
        # 자기보고가 틀린 사람을 20% 심어둔다. 보정 루프가 이걸 찾아내는지가
        # 곧 "설문은 1회차에만 맞다"는 주장의 검증이다.
        true = dict(self_report)
        if rng.random() < 0.35:
            ax = rng.choice(AXES)
            shift = rng.choice([-2, -1, -1, 1, 1, 2])
            true[ax] = max(1, min(5, true[ax] + shift))
            p["_mismatch"] = ax
        p["_true"] = true
        # 이 사람이 실제로 편안해하는 '그룹 다양성 수준'.
        # beta(목적함수 가중치)와는 다른 개념이다. diversity()가 돌려주는
        # 값과 같은 척도(0~1)여야 q4 신호가 의미를 갖는다.
        # 설문에 쓴 관심사와 '실제로 말이 터지는' 테마가 다른 사람 35%.
        # 리뷰의 worked_topics 가 이걸 드러내야 한다.
        if rng.random() < 0.35:
            p["_true_theme"] = rng.choice([t for t in THEME_NAMES if t != main])
            p["_topic_mismatch"] = True
        else:
            p["_true_theme"] = main
        # 실제로 말이 터지는 주제들 (숨은 정답)
        p["_true_interests"] = rng.sample(THEMES[p["_true_theme"]], 3)
        p["_true_tags"] = [p["_true_theme"]]

        p["_div_ideal"] = round(
            max(0.45, min(1.0, rng.gauss(0.82, 0.12))), 3)
        p["confidence"] = round(rng.uniform(0.55, 0.95), 2)
        p["beta"] = BETA_INIT
        profiles.append(p)
    return profiles


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ps = gen(n)
    bad = [(p["user_id"], validate_profile(p)) for p in ps]
    bad = [b for b in bad if b[1]]
    if bad:
        sys.stderr.write("VALIDATION FAILED: %r\n" % bad[:5])
        sys.exit(1)
    sys.stderr.write("generated %d profiles, all valid\n" % len(ps))
    print(json.dumps(ps, ensure_ascii=False, indent=2))
