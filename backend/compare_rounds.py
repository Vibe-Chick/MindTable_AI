"""회차 간 변화 비교 — 루프가 실제로 도는지의 증거.

    python3 compare_rounds.py 1 2
"""
import json
import sys
from itertools import combinations

a, b = int(sys.argv[1]), int(sys.argv[2])
ga = json.load(open("out/groups_real_r%d.json" % a, encoding="utf-8"))
gb = json.load(open("out/groups_real_r%d.json" % b, encoding="utf-8"))
pa = {p["user_id"]: p for p in
      json.load(open("data/profiles_before_r%d.json" % a, encoding="utf-8"))}
pb = {p["user_id"]: p for p in
      json.load(open("data/profiles.json", encoding="utf-8"))}
name = {u: p.get("name", u) for u, p in pb.items()}


def pairs(g):
    return {frozenset(x) for grp in g["groups"] for x in combinations(grp, 2)}


pa_, pb_ = pairs(ga), pairs(gb)
print("=" * 62)
print("%d회차 → %d회차 편성 변화" % (a, b))
print("=" * 62)
for tag, g in ((a, ga), (b, gb)):
    print("\n[%d회차]  목적함수 %.3f" % (tag, g["objective"]["local_search"]))
    for grp in g["groups"]:
        print("   " + " · ".join(name.get(u, u) for u in grp))

keep = pa_ & pb_
print("\n같은 자리 쌍 %d개 중 %d개가 재배치됨 (유지 %d)"
      % (len(pa_), len(pa_ - pb_), len(keep)))
if keep:
    print("유지된 쌍:", ", ".join("%s-%s" % tuple(name.get(u, u) for u in k)
                                for k in keep))

print("\n" + "=" * 62)
print("프로필 이동 (자기보고 → 현재)")
print("=" * 62)
from importlib import import_module
sys.path.insert(0, "src")
sch = import_module("schema")
rows = []
for u, p in pb.items():
    eff = sch.effective(p)
    for ax in sch.AXES:
        d = eff[ax] - p["self_report"][ax]
        if abs(d) >= 0.3:
            rows.append((abs(d), u, ax, p["self_report"][ax], eff[ax], d))
for _, u, ax, sr, ef, d in sorted(rows, reverse=True)[:8]:
    print("  %-6s %s  자기보고 %d → %.2f  (%+.2f)"
          % (name.get(u, u), sch.AXIS_KR[ax], sr, ef, d))

print("\n태그 가중 변화")
for u, p in pb.items():
    w0 = {t: 1.0 for t in (pa[u].get("tags") or [])}
    w1 = {k: round(v, 2) for k, v in (p.get("interest_weights") or {}).items()}
    if w0 != {k: round(v, 2) for k, v in w1.items()}:
        print("  %-6s %r → %r" % (name.get(u, u), w0, w1))
