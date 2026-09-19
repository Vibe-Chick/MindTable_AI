"""2~3회차 시뮬레이션 + 루프 검증.

목적은 두 가지다.

1. 발표용 회차별 곡선 생성 (1회차만 실측, 이후 시뮬레이션 — 슬라이드에 명시할 것)
2. **루프가 실제로 수렴하는지 검증.** mock 프로필에는 숨은 정답(_true)이 심어져
   있고 20%는 자기보고가 틀려 있다. 보정 루프가 그 오차를 줄여나가면
   "설문은 1회차에만 맞다"는 주장이 숫자로 증명된다.

시뮬레이션 회차는 LLM을 쓰지 않는다. 숨은 정답에서 델타를 합성한다.
적용 로직(review.apply_delta)은 실측 경로와 완전히 동일하다.

    python3 src/simulate.py data/profiles_mock.json 3
"""
import copy
import json
import random
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import embed                                                  # noqa: E402
import match                                                  # noqa: E402
import review                                                 # noqa: E402
from schema import (AXES, DELTA_DEADBAND, LONG_TO_AXIS,        # noqa: E402
                    effective)

STEPS = (-1, -0.5, 0, 0.5, 1)
OBS_GAIN = 0.6      # 관측 민감도
OBS_NOISE = 0.35    # 리뷰 한 건에서 성격을 읽는 건 원래 잡음이 크다


def quantize(x):
    """데드밴드 밖의 신호만 통과시킨다. 1회차 잡음에 반응하지 않기 위함."""
    if abs(x) < DELTA_DEADBAND:
        return 0.0
    return min(STEPS, key=lambda s: abs(s - x))


def synth_topics(p, rng):
    """리뷰에서 나올 '실제로 통한 태그 / 죽은 태그'를 만든다."""
    true_tags = p.get("_true_tags", [])
    worked, dead = [], []
    for t in true_tags:
        if rng.random() < 0.6:
            worked.append(t)
    w = p.get("interest_weights") or {}
    stale = [t for t, v in sorted(w.items(), key=lambda kv: -kv[1])
             if t not in true_tags]
    if stale and rng.random() < 0.35:
        dead.append(stale[0])
    if rng.random() < 0.15 and w:          # 오관측
        worked.append(rng.choice(list(w)))
    return worked[:3], dead[:3]


def synth_delta(p, rng):
    """숨은 정답과 현재 벡터의 차이를 잡음 섞어 관측한 것처럼 만든다.

    confidence를 신호 세기에 비례시킨다. 실제 파이프라인에서도 LLM이
    "재밌었어요"뿐인 후기에는 낮은 confidence를 준다 — 약한 신호가 프로필을
    못 움직이게 하는 게 잡음 누적을 막는 핵심이다.
    """
    eff = effective(p)
    out, strength = {}, 0.0
    for long, ax in LONG_TO_AXIS.items():
        gap = p["_true"][ax] - eff[ax]
        raw = gap * OBS_GAIN + rng.gauss(0, OBS_NOISE)
        out[long] = quantize(raw)
        strength = max(strength, abs(raw))
    worked, dead = synth_topics(p, rng)
    out["worked_topics"] = worked
    out["dead_topics"] = dead
    out["evidence"] = "(simulated)"
    out["confidence"] = round(min(1.0, strength), 3)
    return out


def synth_q4(p, group_members, rng):
    """그룹의 실제 다양성이 본인이 편한 수준보다 높았나 낮았나."""
    d = match.diversity(group_members)
    ideal = p["_div_ideal"]
    if rng.random() < 0.15:
        return review.Q4_SAME                 # 응답 잡음
    if d > ideal + 0.08:
        return review.Q4_MORE_SIMILAR
    if d < ideal - 0.08:
        return review.Q4_MORE_DIFFERENT
    return review.Q4_SAME


def _err(p):
    eff = effective(p)
    return sum(abs(eff[a] - p["_true"][a]) for a in AXES) / len(AXES)


def topic_error(profiles):
    """학습된 태그 가중이 '실제로 통하는 태그'와 얼마나 어긋나 있나."""
    tot = 0.0
    for p in profiles:
        true_kw = set(p.get("_true_tags") or [])
        w = p.get("interest_weights") or {}
        s_all = sum(w.values()) or 1.0
        hit = sum(v for k, v in w.items() if k in true_kw)
        tot += 1.0 - hit / s_all
    return tot / len(profiles)


def truth_error(profiles, only=None):
    """현재 벡터가 숨은 정답에서 얼마나 떨어져 있나 (낮을수록 좋다).

    only="mismatch": 자기보고가 틀렸던 사람만 — 제품이 고치겠다고 주장하는 집단
    only="ok":       자기보고가 맞았던 사람만 — 망가뜨리면 안 되는 집단
    """
    ps = profiles
    if only == "mismatch":
        ps = [p for p in profiles if "_mismatch" in p]
    elif only == "ok":
        ps = [p for p in profiles if "_mismatch" not in p]
    if not ps:
        return 0.0
    return sum(_err(p) for p in ps) / len(ps)


def true_sim_matrix(profiles):
    """숨은 정답 관심사로 만든 유사도 행렬. 평가 전용, 회차와 무관하게 고정."""
    shadow = []
    for p in profiles:
        q = copy.deepcopy(p)
        tg = p.get("_true_tags") or p.get("tags") or []
        q["interest_weights"] = {t: 1.0 for t in tg}
        q["tags"] = list(tg)
        q["interest_vec"] = []
        shadow.append(q)
    return embed.similarity_matrix(shadow, mode="tags")


def true_satisfaction(groups, profiles, sim):
    """실제로 좋은 자리였나 — 숨은 정답만으로 계산한다.

    주의: 예전 구현은 shadow 의 beta(학습된 값)를 가중치로 썼다. 그러면 보정으로
    beta 가 내려간 쪽이 자동으로 낮은 점수를 받아, 대조 실험이 성립하지 않았다.
    지표는 두 조건에서 동일해야 하므로 _true / _div_ideal 만 쓴다.

      품질 = 관심사 유사도 + 성격 보완도(정답 벡터) + 다양성 적합도
      다양성 적합도 = 1 - |실제 다양성 - 본인이 편한 수준|
    """
    shadow = []
    for p in profiles:
        q = copy.deepcopy(p)
        q["self_report"] = dict(p["_true"])
        q["behavior"] = {a: 0.0 for a in AXES}
        shadow.append(q)

    tot = 0.0
    for g in groups:
        members = [shadow[i] for i in g]
        pairs = [(g[a], g[b]) for a in range(len(g))
                 for b in range(a + 1, len(g))]
        interest = sum(sim[i][j] for i, j in pairs) / len(pairs)
        d = match.diversity(members)  # noqa
        fit = sum(1.0 - abs(d - profiles[i]["_div_ideal"]) for i in g) / len(g)
        tot += interest + match.GAMMA * match.complement(members) + fit
    return tot / len(groups)


def run(profiles, rounds=3, seed=1, correct=True):
    """correct=False 면 델타를 적용하지 않는다 (대조군)."""
    rng = random.Random(seed)
    tsim = true_sim_matrix(profiles)      # 평가용, 고정
    history, snapshots = [], []

    for r in range(1, rounds + 1):
        # 관심사 가중이 회차마다 바뀌므로 유사도를 매번 다시 만든다
        sim = embed.similarity_matrix(profiles)
        groups, leftover = match.greedy(profiles, sim, explore=True, seed=r)
        groups, obj, _ = match.local_search(groups, profiles, sim)

        snapshots.append(copy.deepcopy(profiles))
        history.append({
            "round": r,
            "objective": round(obj, 3),
            "true_satisfaction": round(true_satisfaction(groups, profiles,
                                                         tsim), 4),
            "err_all": round(truth_error(profiles), 4),
            "err_mismatch": round(truth_error(profiles, "mismatch"), 4),
            "err_ok": round(truth_error(profiles, "ok"), 4),
            "topic_err": round(topic_error(profiles), 4),
            "leftover": len(leftover),
            "mean_beta": round(sum(p["beta"] for p in profiles)
                               / len(profiles), 3),
        })

        # --- 보정: 리뷰 수거 → 델타 → 적용 ---
        for gi, g in enumerate(groups):
            gid = "g_r%d_%02d" % (r, gi)
            members = [profiles[i] for i in g]
            review.record_meeting(profiles, g, gid)
            for i in g:
                p = profiles[i]
                if rng.random() < 0.12:        # 리뷰 미제출 12%
                    continue
                if not correct:
                    continue
                review.apply_delta(p, synth_delta(p, rng),
                                   synth_q4(p, members, rng))

    history.append({
        "round": rounds + 1, "objective": None, "true_satisfaction": None,
        "err_all": round(truth_error(profiles), 4),
        "err_mismatch": round(truth_error(profiles, "mismatch"), 4),
        "err_ok": round(truth_error(profiles, "ok"), 4),
        "topic_err": round(topic_error(profiles), 4),
        "leftover": None,
        "mean_beta": round(sum(p["beta"] for p in profiles)
                           / len(profiles), 3),
    })
    return history, snapshots, profiles


def ablation(profiles, rounds=3, seed=1):
    """보정 ON/OFF 대조 실험.

    왜 필요한가: 회차별 만족도 곡선은 그냥 내려간다. 60명 고정 풀에서
    반복 매칭을 피하다 보면 좋은 조합이 소진되기 때문이다 (조합 고갈).
    이건 보정의 성능과 무관하다.

    따라서 보정의 가치는 '곡선이 올라가는가'가 아니라
    '같은 회차에서 보정한 쪽이 안 한 쪽보다 나은가'로 측정해야 한다.
    """
    on, _, _ = run(copy.deepcopy(profiles), rounds, seed, correct=True)
    off, _, _ = run(copy.deepcopy(profiles), rounds, seed, correct=False)
    rows = []
    for a, b in zip(on[:-1], off[:-1]):
        gap = a["true_satisfaction"] - b["true_satisfaction"]
        rows.append({
            "round": a["round"],
            "corrected": a["true_satisfaction"],
            "frozen": b["true_satisfaction"],
            "gap": round(gap, 4),
            "gap_pct": round(gap / abs(b["true_satisfaction"]) * 100, 2),
        })
    return rows


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/profiles_mock.json"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    profiles = json.load(open(path, encoding="utf-8"))
    hist, snaps, final = run(profiles, rounds)

    print("round  objective  true_sat  err_mismatch  err_ok  topic_err")
    for h in hist:
        print("  %-4s %9s %9s %13.4f %7.4f %10.4f"
              % (h["round"],
                 "-" if h["objective"] is None else "%.3f" % h["objective"],
                 "-" if h["true_satisfaction"] is None
                 else "%.4f" % h["true_satisfaction"],
                 h["err_mismatch"], h["err_ok"], h["topic_err"]))
    t0, t1 = hist[0]["topic_err"], hist[-1]["topic_err"]
    print("\n관심사 오차  %.4f → %.4f  (%+.1f%%)"
          % (t0, t1, (t1 - t0) / t0 * 100))

    m0, m1 = hist[0]["err_mismatch"], hist[-1]["err_mismatch"]
    o0, o1 = hist[0]["err_ok"], hist[-1]["err_ok"]
    print("\n자기보고 틀렸던 집단  %.4f → %.4f  (%+.1f%%)"
          % (m0, m1, (m1 - m0) / m0 * 100))
    print("자기보고 맞았던 집단  %.4f → %.4f  (잡음 유입량)" % (o0, o1))
    assert m1 < m0, "수렴 실패: 틀린 자기보고를 고치지 못했다"
    assert o1 < 0.25, "잡음 과다: 정확했던 프로필을 망가뜨린다 (%.3f)" % o1
    print("OK: 틀린 프로필은 고치고, 맞았던 프로필은 거의 안 건드린다")

    # 클리핑·범위 검증
    from schema import BEHAVIOR_CLIP, BETA_MAX, BETA_MIN
    for p in final:
        for a in AXES:
            assert abs(p["behavior"][a]) <= BEHAVIOR_CLIP + 1e-9, \
                "behavior 발산: %s %s=%r" % (p["user_id"], a, p["behavior"][a])
        assert BETA_MIN - 1e-9 <= p["beta"] <= BETA_MAX + 1e-9, \
            "beta 범위 이탈: %s=%r" % (p["user_id"], p["beta"])
    print("OK: behavior 클리핑, beta 범위 유지")

    print("\n=== 보정 ON/OFF 대조 (같은 회차 비교) ===")
    print("round  보정함     보정안함   차이      %")
    for r in ablation(json.load(open(path, encoding="utf-8")), rounds):
        print("  %-4d %9.4f %10.4f %+8.4f %+7.2f%%"
              % (r["round"], r["corrected"], r["frozen"],
                 r["gap"], r["gap_pct"]))

    mism = [p for p in final if "_mismatch" in p]
    print("\n자기보고가 틀렸던 %d명 중 상위 5명의 보정 결과:" % len(mism))
    for p in mism[:5]:
        ax = p["_mismatch"]
        print("  %s %s  %s: 자기보고 %d / 실제 %d / 보정후 %.2f"
              % (p["user_id"], p.get("name", ""), ax,
                 p["self_report"][ax], p["_true"][ax], effective(p)[ax]))
