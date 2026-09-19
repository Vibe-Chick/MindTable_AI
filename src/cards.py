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


CORRECTION = """

[재작성 지시] 직전 출력이 규칙을 어겼다: %s
- targets 가 1명뿐인 질문 금지. 1번은 전원, 2·3번은 최소 2명.
- 세 질문의 targets 합집합에 전원이 들어가야 한다.
- 모든 질문에 프로필의 구체적 활동을 그대로 적는다.
  1번은 서로 다른 사람의 활동 2개 이상, 2·3번은 1개 이상.
- 예시 문장을 복사하지 마라. 이 그룹의 프로필에서만 가져와라.
다시 써라."""


def _hits(text, keywords):
    """질문에 프로필의 구체적 활동이 몇 개나 적혀 있는가."""
    t = text.replace(" ", "")
    return sum(1 for k in keywords if k.replace(" ", "") in t)


def _check(out, names, keywords=()):
    """모델 출력을 검증한다. 프롬프트 준수를 믿지 않는다."""
    errs, covered = [], set()
    for i, ib in enumerate(out.get("icebreakers") or []):
        tg = [t for t in (ib.get("targets") or []) if t in names]
        covered |= set(tg)
        q = ib.get("question", "")
        if i == 0 and len(tg) < len(names):
            errs.append("1번이 전원 대상이 아님(%d/%d)" % (len(tg), len(names)))
        elif i > 0 and len(tg) < 2:
            errs.append("%d번이 1인 전용" % (i + 1))
        # 구체성: 프로필의 활동 키워드가 질문에 실제로 적혀 있어야 한다.
        # 없으면 어느 그룹에나 붙는 범용 질문이다.
        need = 2 if i == 0 else 1
        if keywords and _hits(q, keywords) < need:
            errs.append("%d번이 범용 질문(구체적 활동 %d개 미만)"
                        % (i + 1, need))
    miss = names - covered
    if miss:
        errs.append("질문에 안 나오는 사람: %s" % ", ".join(sorted(miss)))
    return errs


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
    names = {p.get("name", p["user_id"]) for p in members}
    kws = [k for p in members for k in p.get("interests", [])]
    out = llm.call(prompt, CARD_SCHEMA, system=prompts.CARD_SYSTEM,
                   temp=0.7, tag="card:" + group_id)
    errs = _check(out, names, kws)
    if errs:
        # 한 번만 다시 묻는다. 프롬프트가 바뀌므로 캐시 키도 바뀐다.
        sys.stderr.write("[cards] %s 재작성: %s\n" % (group_id, "; ".join(errs)))
        out2 = llm.call(prompt + CORRECTION % "; ".join(errs), CARD_SCHEMA,
                        system=prompts.CARD_SYSTEM, temp=0.7,
                        tag="card:%s:retry" % group_id)
        if not _check(out2, names, kws):
            out = out2
        else:
            sys.stderr.write("[cards] %s 재작성도 실패 — 그대로 사용\n"
                             % group_id)
            out = out2

    qs, tgs = [], []
    for ib in out["icebreakers"]:
        qs.append(ib["question"])
        tgs.append([t for t in (ib.get("targets") or []) if t in names])
    return {
        "group_id": group_id,
        "members": [{"user_id": p["user_id"], "name": p.get("name", ""),
                     "school": p["school"], "major": p["major"],
                     "college": p["college"], "year": p["year"],
                     "interests": p["interests"]} for p in members],
        "reason": out["reason"],
        "icebreakers": qs,                 # 프론트엔드 계약: 문자열 배열 유지
        "icebreaker_targets": tgs,
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
