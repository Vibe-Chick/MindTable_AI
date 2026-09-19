"""발표에 쓸 숫자를 한 파일로 모은다. out/metrics.json 이 유일한 출처다.

슬라이드에 손으로 숫자를 옮겨 적지 말 것. 파이프라인을 다시 돌리면 값이 바뀐다.
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from schema import AXES, effective                            # noqa: E402


def reliability(manual_path, profiles):
    """Stage 3 — 팀원 수기 채점 vs LLM 채점의 축별 일치도.

    manual_scores.csv: user_id,O,C,E,A,N
    동일 응답 2회 재채점이 아니라 사람 판단과의 비교여야 한다.
    """
    import csv
    import os
    if not os.path.exists(manual_path):
        return None
    by_id = {p["user_id"]: p for p in profiles}
    pairs = {a: [] for a in AXES}
    with open(manual_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = by_id.get(row["user_id"])
            if not p:
                continue
            for a in AXES:
                if row.get(a):
                    pairs[a].append((int(row[a]), p["self_report"][a]))
    out = {}
    for a, vs in pairs.items():
        if len(vs) < 3:
            out[a] = None
            continue
        n = len(vs)
        mh = sum(h for h, _ in vs) / n
        ml = sum(l for _, l in vs) / n
        cov = sum((h - mh) * (l - ml) for h, l in vs)
        vh = sum((h - mh) ** 2 for h, _ in vs) ** 0.5
        vl = sum((l - ml) ** 2 for _, l in vs) ** 0.5
        out[a] = {
            "n": n,
            "pearson": round(cov / (vh * vl), 3) if vh and vl else None,
            "within_1": round(sum(1 for h, l in vs if abs(h - l) <= 1) / n, 3),
        }
    return out


def collect(history, bench, profiles, cards, manual_path=None):
    div = []
    for p in profiles:
        eff = effective(p)
        worst = max(AXES, key=lambda a: abs(eff[a] - p["self_report"][a]))
        d = eff[worst] - p["self_report"][worst]
        if abs(d) >= 0.5:
            div.append({"user_id": p["user_id"], "name": p.get("name", ""),
                        "axis": worst, "self_report": p["self_report"][worst],
                        "effective": round(eff[worst], 2), "delta": round(d, 2)})
    div.sort(key=lambda r: -abs(r["delta"]))

    return {
        "n_users": len(profiles),
        "n_groups": len(cards),
        "objective": bench["objective"],
        "objective_gain_vs_random_pct": round(
            (bench["objective"]["local_search"] - bench["objective"]["random"])
            / abs(bench["objective"]["random"]) * 100, 1),
        "local_search_iters": bench["local_search_iters"],
        "carry_over": bench["leftover"],
        "rounds": history,
        "satisfaction_curve": [h["true_satisfaction"] for h in history
                               if h["true_satisfaction"] is not None],
        "divergence_top": div[:10],
        "reliability": (reliability(manual_path, profiles)
                        if manual_path else None),
        "cards_without_llm": sum(1 for c in cards if not c.get("reason")),
    }


def write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
