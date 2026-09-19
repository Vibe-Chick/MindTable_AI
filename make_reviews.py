"""리뷰 입력 템플릿 생성. 사람이 채워 넣을 파일을 만든다.

    python3 make_reviews.py 1     → data/reviews_r1.json (빈 템플릿)

_ 로 시작하는 키는 작성 참고용이며 파이프라인이 무시한다.
"""
import json
import os
import sys

rnd = int(sys.argv[1]) if len(sys.argv) > 1 else 1
src = sys.argv[2] if len(sys.argv) > 2 else "out/groups_real.json"
out = "data/reviews_r%d.json" % rnd

g = json.load(open(src, encoding="utf-8"))
profiles = {p["user_id"]: p
            for p in json.load(open("data/profiles.json", encoding="utf-8"))}

if os.path.exists(out):
    sys.exit("이미 있다: %s (덮어쓰려면 먼저 지워라)" % out)

tpl = {}
for card in g["cards"]:
    ids = [m["user_id"] for m in card["members"]]
    for m in card["members"]:
        mates = [x["name"] for x in card["members"]
                 if x["user_id"] != m["user_id"]]
        tpl[m["user_id"]] = {
            "_이름": m["name"],
            "_그룹": card["group_id"],
            "_같은자리": mates,
            "_내관심사": m["interests"],
            "_받았던_대화주제": card["icebreakers"],
            "r1": "",
            "r2": "",
            "r3": "",
            "q4": "지금 정도",
        }

json.dump(tpl, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("생성: %s  (%d명)" % (out, len(tpl)))
print("""
채울 것:
  r1  대화가 제일 잘 풀린 순간이 언제였나요? 무슨 얘기 중이었나요?
  r2  말이 끊기거나 어색했던 주제가 있었나요?
  r3  본인은 주로 이끄는 쪽이었나요, 듣는 쪽이었나요?
  q4  "더 비슷" | "지금 정도" | "더 달라도 됨"

비워두면 미제출 처리된다 (델타 0). 일부만 채워도 된다.
""")
