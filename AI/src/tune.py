"""파라미터 그리드 서치.

숨은 정답 시뮬레이션을 목적함수로 쓴다.
  최대화: 자기보고가 틀렸던 집단의 오차 감소량
  제약:   자기보고가 맞았던 집단의 오차 <= 0.15 (망가뜨리면 안 된다)

    python3 src/tune.py
"""
import copy
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import schema                                                # noqa: E402
import simulate                                              # noqa: E402

GRID = [(lr, dc, db)
        for lr in (0.4, 0.5, 0.7, 0.9)
        for dc in (0.10, 0.15, 0.25)
        for db in (0.35, 0.45, 0.60)]
OK_LIMIT = 0.15


def trial(base, lr, dc, db, rounds=5):
    schema.LR = lr
    schema.BEHAVIOR_DECAY = dc
    schema.DELTA_DEADBAND = db
    import review
    review.LR, review.BEHAVIOR_DECAY = lr, dc
    simulate.DELTA_DEADBAND = db
    hist, _, _ = simulate.run(copy.deepcopy(base), rounds=rounds, seed=1)
    return hist[0], hist[-1]


if __name__ == "__main__":
    base = json.load(open("data/profiles_mock.json", encoding="utf-8"))
    rows = []
    for lr, dc, db in GRID:
        a, b = trial(base, lr, dc, db)
        gain = (a["err_mismatch"] - b["err_mismatch"]) / a["err_mismatch"] * 100
        rows.append({"lr": lr, "decay": dc, "deadband": db,
                     "gain_pct": round(gain, 1),
                     "err_ok": b["err_ok"],
                     "sat_delta": round(b["true_satisfaction"] or 0, 4)})
    ok = [r for r in rows if r["err_ok"] <= OK_LIMIT]
    ok.sort(key=lambda r: -r["gain_pct"])
    print("제약 통과 %d/%d  (err_ok <= %.2f)\n" % (len(ok), len(rows), OK_LIMIT))
    print(" LR  decay deadband   개선율  err_ok")
    for r in ok[:10]:
        print("%.1f  %.2f    %.2f    %+6.1f%%  %.4f"
              % (r["lr"], r["decay"], r["deadband"],
                 r["gain_pct"], r["err_ok"]))
    if ok:
        b = ok[0]
        print("\n권장: LR=%.1f  BEHAVIOR_DECAY=%.2f  DELTA_DEADBAND=%.2f"
              % (b["lr"], b["decay"], b["deadband"]))
