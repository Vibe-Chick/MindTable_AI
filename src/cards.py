"""Stage 6 — 그룹 카드 생성. 세 번째 LLM 호출 지점 (그룹당 1콜).

식당 추천은 제휴 DB가 없으므로 캠퍼스 상권 하드코딩 리스트에서 고른다.
해커톤 기간에 제휴는 못 딴다. 데모는 이걸로 충분하다.
"""
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import llm                                                    # noqa: E402
import prompts                                                # noqa: E402
from schema import CARD_SCHEMA                                 # noqa: E402

RESTAURANTS = [
    {"name": "상도동 정육식당", "category": "고기", "price": "1.5만",
     "near": "숭실대"},
    {"name": "흑석 파스타바", "category": "양식", "price": "1.3만",
     "near": "중앙대"},
    {"name": "왕십리 국밥집", "category": "한식", "price": "0.9만",
     "near": "한양대"},
    {"name": "건대 양꼬치거리", "category": "중식", "price": "1.4만",
     "near": "건국대"},
    {"name": "공릉 카레하우스", "category": "일식", "price": "1.1만",
     "near": "서울여대"},
    {"name": "충무로 노포 분식", "category": "분식", "price": "0.7만",
     "near": "동국대"},
]


def pick_restaurant(members):
    """가장 많은 구성원이 가까운 상권. 동률이면 가장 싼 곳."""
    schools = Counter(p["school"] for p in members)
    cands = [r for r in RESTAURANTS if r["near"] in schools]
    if not cands:
        cands = RESTAURANTS
    return max(cands, key=lambda r: (schools.get(r["near"], 0),
                                     -float(r["price"].rstrip("만"))))


def overlap_keywords(members):
    c = Counter(k for p in members for k in p["interests"])
    return [k for k, n in c.items() if n >= 2] or []


def build_card(members, group_id):
    lines = []
    for p in members:
        lines.append("- %s (%s %s %d학년) 관심사: %s"
                     % (p.get("name", p["user_id"]), p["school"], p["major"],
                        p["year"], ", ".join(p["interests"])))
    ov = overlap_keywords(members)
    prompt = prompts.CARD_USER.format(
        members="\n".join(lines),
        overlap=", ".join(ov) if ov else "(직접 겹치는 키워드 없음)",
        majors=", ".join(sorted({p["major"] for p in members})))
    out = llm.call(prompt, CARD_SCHEMA, system=prompts.CARD_SYSTEM,
                   temp=0.7, tag="card:" + group_id)
    return {
        "group_id": group_id,
        "members": [{"user_id": p["user_id"], "name": p.get("name", ""),
                     "school": p["school"], "major": p["major"],
                     "college": p["college"], "year": p["year"],
                     "interests": p["interests"]} for p in members],
        "reason": out["reason"],
        "icebreakers": out["icebreakers"],
        "overlap": ov,
        "restaurant": pick_restaurant(members),
    }


def build_all(groups, profiles, round_no):
    cards = []
    for gi, g in enumerate(groups):
        gid = "g_r%d_%02d" % (round_no, gi)
        members = [profiles[i] for i in g]
        try:
            cards.append(build_card(members, gid))
        except Exception as e:
            sys.stderr.write("[cards] %s 실패: %r\n" % (gid, e))
            cards.append({
                "group_id": gid,
                "members": [{"user_id": p["user_id"], "name": p.get("name", ""),
                             "school": p["school"], "major": p["major"],
                             "college": p["college"], "year": p["year"],
                             "interests": p["interests"]} for p in members],
                "reason": None, "icebreakers": [],
                "overlap": overlap_keywords(members),
                "restaurant": pick_restaurant(members),
            })
    return cards
