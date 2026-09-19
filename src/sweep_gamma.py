"""성격 가중치(GAMMA)에 따라 보정이 값을 내는지 확인.

배경: 보정 루프는 프로필 오차를 20% 줄이지만 매칭 품질은 거의 안 바뀐다.
목적함수 3개 항 중 성격(complement)만 프로필에 의존하고, 나머지 둘
(관심사 유사도·전공 다양성)은 보정과 무관한 고정값이기 때문이다.

즉 보정의 가치는 GAMMA 에 비례한다. 어느 지점부터 값이 나오는지 본다.
"""
import copy
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import match                                                  # noqa: E402
import simulate                                               # noqa: E402

if __name__ == "__main__":
    base = json.load(open("data/profiles_mock.json", encoding="utf-8"))
    print("GAMMA  마지막회차 보정함  보정안함     차이       %")
    for g in (0.5, 1.0, 1.5, 2.0, 3.0):
        match.GAMMA = g          # 최적화기와 평가지표에 동시 적용
        rows = simulate.ablation(copy.deepcopy(base), rounds=4)
        r = rows[-1]
        print("%5.1f %14.4f %10.4f %+9.4f %+7.2f%%"
              % (g, r["corrected"], r["frozen"], r["gap"], r["gap_pct"]))
