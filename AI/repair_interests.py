"""표시용 키워드 복구.

apply_topics 버그로 interests(자유 키워드)가 태그 이름으로 덮어써졌다.
추출 직후 스냅샷에서 되살린다. 학습된 가중(interest_weights)과 만남
이력(met/history)은 그대로 둔다.
"""
import json
import os
import sys

cur = json.load(open("data/profiles.json", encoding="utf-8"))
src = None
for cand in ("data/profiles_before_r1.json",):
    if os.path.exists(cand):
        src = {p["user_id"]: p for p in json.load(open(cand, encoding="utf-8"))}
        break
if not src:
    sys.exit("복구 원본이 없다: data/profiles_before_r1.json")

n = 0
for p in cur:
    o = src.get(p["user_id"])
    if not o:
        continue
    if p.get("interests") != o.get("interests"):
        p["interests"] = list(o["interests"])
        p["interest_evidence"] = o.get("interest_evidence", {})
        n += 1

json.dump(cur, open("data/profiles.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("복구 %d명" % n)
for p in cur:
    print("  %-6s 키워드 %-32s 태그 %r"
          % (p.get("name", ""), ", ".join(p["interests"]),
             {k: round(v, 2) for k, v in
              sorted((p.get("interest_weights") or {}).items(),
                     key=lambda kv: -kv[1])}))
