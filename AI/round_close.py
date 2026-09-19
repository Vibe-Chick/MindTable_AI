"""한 회차를 마감한다: 리뷰 → 델타 → 프로필 갱신 → 다음 회차 편성.

    source .env && python3 round_close.py 1

하는 일
  1. reviews_r{n}.json 을 LLM 에 넣어 축별 델타 + 태그 신호 추출
  2. apply_delta 로 프로필 갱신 (감쇠·클리핑·confidence 가중)
  3. 같은 자리에 앉은 쌍을 기록 (다음 회차 반복 페널티)
  4. 변화 전/후를 출력
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import llm                                                    # noqa: E402
import review                                                 # noqa: E402
from schema import AXES, effective                            # noqa: E402

rnd = int(sys.argv[1]) if len(sys.argv) > 1 else 1
profiles = json.load(open("data/profiles.json", encoding="utf-8"))
reviews = json.load(open("data/reviews_r%d.json" % rnd, encoding="utf-8"))
groups = json.load(open("out/groups_real.json", encoding="utf-8"))
by_id = {p["user_id"]: p for p in profiles}
idx = {p["user_id"]: i for i, p in enumerate(profiles)}

# 회차 전 스냅샷. 다시 돌리려면 이 파일을 profiles.json 으로 되돌린다.
bak = "data/profiles_before_r%d.json" % rnd
if not os.path.exists(bak):
    json.dump(profiles, open(bak, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("회차 전 백업: %s" % bak)

before = copy.deepcopy(by_id)
done = skipped = 0

for uid, rv in reviews.items():
    p = by_id.get(uid)
    if p is None:
        continue
    if not any((rv.get(k) or "").strip() for k in ("r1", "r2", "r3")):
        skipped += 1
        continue
    try:
        d = review.extract_delta(p, rv)
    except Exception as e:
        sys.stderr.write("[%s] 델타 추출 실패 → 0 처리: %r\n" % (uid, e))
        skipped += 1
        continue
    review.apply_delta(p, d, rv.get("q4"))
    p.setdefault("delta_log", []).append({"round": rnd, "delta": d})
    done += 1

# 같은 자리 기록
for card in groups["cards"]:
    ids = [m["user_id"] for m in card["members"] if m["user_id"] in idx]
    review.record_meeting(profiles, [idx[u] for u in ids], card["group_id"])

json.dump(profiles, open("data/profiles.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
json.dump(profiles, open("data/profiles_r%d.json" % rnd, "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)

print("=" * 66)
print("%d회차 마감 — 반영 %d명 / 미제출 %d명" % (rnd, done, skipped))
print("=" * 66)
for p in profiles:
    b, a = before[p["user_id"]], p
    eb, ea = effective(b), effective(a)
    ch = [(x, eb[x], ea[x]) for x in AXES if abs(ea[x] - eb[x]) > 0.01]
    tb = {k: round(v, 2) for k, v in (b.get("interest_weights") or {}).items()}
    ta = {k: round(v, 2) for k, v in (a.get("interest_weights") or {}).items()}
    if not ch and tb == ta:
        continue
    print("\n%s %s" % (p["user_id"], p.get("name", "")))
    for x, v0, v1 in ch:
        print("   %s  %.2f → %.2f  (자기보고 %d)"
              % (x, v0, v1, p["self_report"][x]))
    if tb != ta:
        print("   태그 %r → %r" % (tb, ta))
    lg = (p.get("delta_log") or [{}])[-1].get("delta", {})
    if lg.get("evidence"):
        print("   근거: %s" % lg["evidence"][:90])

print("\n캐시:", llm.cache_stats())
print("\n다음: python3 run_real.py  (갱신된 프로필로 %d회차 편성)" % (rnd + 1))
