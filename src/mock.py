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
from schema import AXES, BETA_INIT, new_profile, validate_profile  # noqa: E402

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

FIRST = ["지원", "서연", "민준", "하은", "도윤", "수빈", "예준", "지우", "현우",
         "다은", "준서", "소율", "시우", "유진", "건우", "채원", "태윤", "나연"]


def gen(n=60, seed=42):
    rng = random.Random(seed)
    profiles = []
    for i in range(n):
        school = rng.choice(list(SCHOOLS))
        college, major = rng.choice(SCHOOLS[school])

        # 주 테마 2개에서 관심사를 뽑는다 (교집합이 생기도록)
        main, sub = rng.sample(THEME_NAMES, 2)
        interests = rng.sample(THEMES[main], 2) + [rng.choice(THEMES[sub])]

        # 성격: 정규분포 비슷하게, 1~5 정수
        self_report = {}
        for a in AXES:
            v = round(rng.gauss(3.0, 0.9))
            self_report[a] = max(1, min(5, v))

        p = new_profile(
            user_id="u%03d" % (i + 1),
            school=school, major=major, college=college,
            year=rng.choice([1, 1, 2, 2, 3, 4]),
            self_report=self_report,
            interests=interests,
        )
        p["name"] = rng.choice(FIRST)          # 데모 화면용 표시명
        p["theme"] = main                      # 검증용. 실제 파이프라인엔 없음
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
