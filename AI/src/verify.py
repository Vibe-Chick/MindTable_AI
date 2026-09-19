"""시드 여러 개로 결과가 재현되는지 확인. 단일 시드 결과는 믿지 않는다."""
import copy
import json
import statistics as st
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import mock                                                   # noqa: E402
import simulate                                               # noqa: E402

if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    gaps, topics, oks = [], [], []
    print("seed   보정ON    보정OFF    차이%   관심사오차감소%   err_ok")
    for seed in range(1, 9):
        base = mock.gen(60, seed=100 + seed)
        rows = simulate.ablation(copy.deepcopy(base), rounds=rounds, seed=seed)
        hist, _, _ = simulate.run(copy.deepcopy(base), rounds=rounds, seed=seed)
        r = rows[-1]
        t = (hist[-1]["topic_err"] - hist[0]["topic_err"]) / hist[0]["topic_err"] * 100
        gaps.append(r["gap_pct"]); topics.append(t); oks.append(hist[-1]["err_ok"])
        print("%4d %8.4f %9.4f %+8.2f %14.1f %9.4f"
              % (seed, r["corrected"], r["frozen"], r["gap_pct"], t,
                 hist[-1]["err_ok"]))
    print("\n매칭 품질 차이  평균 %+.2f%%  (중앙값 %+.2f%%, 최소 %+.2f%%, 최대 %+.2f%%)"
          % (st.mean(gaps), st.median(gaps), min(gaps), max(gaps)))
    print("관심사 오차감소 평균 %.1f%%" % st.mean(topics))
    print("정확했던 프로필 잡음 평균 %.4f (최대 %.4f)" % (st.mean(oks), max(oks)))
    assert st.median(gaps) > 5, "중앙값이 5%% 미만: 효과가 불안정하다"
    assert min(gaps) > 0, "일부 시드에서 보정이 역효과"
    print("\nOK: 8개 시드 전부에서 보정이 매칭을 개선한다")
