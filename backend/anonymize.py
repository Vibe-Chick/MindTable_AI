"""저장소에 올릴 샘플을 만든다. 이름과 연락처를 제거한다.

실제 응답이 들어오면 data/ 와 out/groups_real*.json 에는 학생들의
자유서술과 이름이 들어간다. 그건 커밋하지 않는다 (.gitignore).
프론트엔드가 쓸 구조 샘플만 익명화해서 out/sample/ 로 내보낸다.

    python3 anonymize.py
"""
import json
import os
import re

SRC_PROFILES = "data/profiles.json"
SRC_GROUPS = "out/groups_real.json"
DST = "out/sample"
DROP = ("name", "evidence", "interest_evidence", "delta_log",
        "dropped", "contact", "연락처")

os.makedirs(DST, exist_ok=True)
profiles = json.load(open(SRC_PROFILES, encoding="utf-8"))

alias = {}
for i, p in enumerate(profiles, 1):
    if p.get("name"):
        alias[p["name"]] = "학생%02d" % i

def scrub_text(t):
    for real, fake in alias.items():
        t = re.sub(re.escape(real) + r"(이가|이는|이랑|이|가|는|은|와|과|의|도)?",
                   fake, t)
    return t

def scrub_profile(p, i):
    q = {k: v for k, v in p.items()
         if k not in DROP and not k.startswith("_")}
    q["user_id"] = "u%03d" % i
    q["display"] = alias.get(p.get("name", ""), "학생%02d" % i)
    return q

out_p = [scrub_profile(p, i) for i, p in enumerate(profiles, 1)]
json.dump(out_p, open(os.path.join(DST, "profiles.json"), "w",
                      encoding="utf-8"), ensure_ascii=False, indent=2)

if os.path.exists(SRC_GROUPS):
    g = json.load(open(SRC_GROUPS, encoding="utf-8"))
    for card in g.get("cards", []):
        for m in card.get("members", []):
            m["display"] = alias.get(m.pop("name", ""), "학생")
        card["reason"] = scrub_text(card.get("reason") or "")
        card["icebreakers"] = [scrub_text(q) for q in card.get("icebreakers", [])]
        card["icebreaker_targets"] = [
            [alias.get(t, t) for t in tg]
            for tg in card.get("icebreaker_targets", [])]
    json.dump(g, open(os.path.join(DST, "groups.json"), "w",
                      encoding="utf-8"), ensure_ascii=False, indent=2)

print("익명화 완료 → %s/" % DST)
print("  %d명, 이름 %d개 치환" % (len(out_p), len(alias)))
