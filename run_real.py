"""실제 프로필로 편성 + 카드까지. 시뮬레이션 없음.

    source .env && python3 run_real.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import cards      # noqa: E402
import embed      # noqa: E402
import llm        # noqa: E402
import match      # noqa: E402

RND = int(sys.argv[1]) if len(sys.argv) > 1 else 1
profiles = json.load(open("data/profiles.json", encoding="utf-8"))
print("%d회차 편성 — 프로필 %d건" % (RND, len(profiles)))

if os.environ.get("EMBED_MODEL"):
    try:
        embed.from_api(profiles)
        print("임베딩 적용")
    except Exception as e:
        print("임베딩 생략:", e)

sim = embed.similarity_matrix(profiles)
b = match.benchmark(profiles, sim)
o = b["objective"]
print("편성  random=%.3f  greedy=%.3f  local=%.3f  (그룹 %d, 이월 %d)"
      % (o["random"], o["greedy"], o["local_search"],
         len(b["groups"]), b["leftover"]))

cs = cards.build_all(b["groups"], profiles, RND)
os.makedirs("out", exist_ok=True)
payload = {"round": RND,
           "groups": [[profiles[i]["user_id"] for i in g] for g in b["groups"]],
           "objective": o, "cards": cs}
for path in ("out/groups_real.json", "out/groups_real_r%d.json" % RND):
    json.dump(payload, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

for c in cs:
    print("\n" + "=" * 64)
    print(c["group_id"], "|", c["restaurant"]["name"], c["restaurant"]["price"])
    for m in c["members"]:
        print("  %s (%s %s) %s" % (m["name"], m["school"], m["major"],
                                   ", ".join(m["interests"])))
    print("  이유:", c["reason"])
    for q in c["icebreakers"]:
        print("   -", q)
print("\n캐시:", llm.cache_stats())
