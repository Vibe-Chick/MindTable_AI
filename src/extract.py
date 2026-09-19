"""Stage 2 — 설문 응답 → 프로필 벡터.

첫 번째 LLM 호출 지점. 유저당 1콜 (문항별로 4콜 하지 않는다: 문맥이 끊기면
채점이 흔들리고 비용이 4배다).

    python3 src/extract.py data/responses.csv --dry-run   # 프롬프트만 확인
    python3 src/extract.py data/responses.csv --one       # 1건만 실제 호출
    python3 src/extract.py data/responses.csv             # 전체
"""
import csv
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import llm                                                    # noqa: E402
import prompts                                                # noqa: E402
from schema import (EXTRACTION_SCHEMA, LONG_TO_AXIS, TAGS,     # noqa: E402
                    new_profile, validate_profile)

# 구글폼 export 헤더 → 내부 키. 폼 확정되면 여기만 고친다.
COLMAP = {
    "school": "학교", "major": "전공", "college": "단과대", "year": "학년",
    "name": "이름",
    "a1": "Q1", "a2": "Q2", "a3": "Q3", "a4": "Q4",
}


def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("빈 CSV")
    missing = [v for v in COLMAP.values() if v not in rows[0]]
    if missing:
        raise SystemExit(
            "CSV 헤더 불일치: %r\n실제 헤더: %r\n"
            "→ src/extract.py 의 COLMAP 을 폼에 맞게 고칠 것."
            % (missing, list(rows[0])))
    return rows


def _norm(t):
    """공백·줄바꿈을 지워서 인용 대조용으로 정규화."""
    return "".join((t or "").split())


def source_text(row):
    return " ".join(row[COLMAP[k]] or "" for k in ("a1", "a2", "a3", "a4"))


def build_prompt(row):
    return prompts.EXTRACT_USER.format(
        a1=row[COLMAP["a1"]].strip() or "(무응답)",
        a2=row[COLMAP["a2"]].strip() or "(무응답)",
        a3=row[COLMAP["a3"]].strip() or "(무응답)",
        a4=row[COLMAP["a4"]].strip() or "(무응답)")


def extract_one(row, uid):
    out = llm.call(build_prompt(row), EXTRACTION_SCHEMA,
                   system=prompts.EXTRACT_SYSTEM, temp=0.0,
                   tag="extract:" + uid)
    src = _norm(source_text(row))
    scores, evidence, dropped = {}, {}, []

    for long, ax in LONG_TO_AXIS.items():
        quote = (out.get(long + "_quote") or "").strip()
        sc = int(out[long])
        # 인용이 비었거나 답변 원문에 실제로 없으면 점수를 버린다.
        # 모델은 요약을 인용이라고 내놓는 경향이 있는데, 그건 근거가 아니다.
        grounded = bool(quote) and _norm(quote) in src
        if not grounded:
            if sc != 3:
                dropped.append("%s:%d→3%s" % (ax, sc, "" if not quote
                                              else "(인용불일치)"))
            sc, quote = 3, ""
        scores[ax] = sc
        evidence[ax] = quote

    kws, kq = out.get("interests") or [], out.get("interest_quotes") or []
    interests, kw_ev = [], {}
    for i, k in enumerate(kws):
        k = (k or "").strip()
        q = (kq[i] if i < len(kq) else "").strip()
        if not k:
            continue
        if not q or _norm(q) not in src:
            dropped.append("관심사 '%s' 근거없음→제거" % k)
            continue
        interests.append(k)
        kw_ev[k] = q

    p = new_profile(
        user_id=uid,
        school=row[COLMAP["school"]].strip(),
        major=row[COLMAP["major"]].strip(),
        college=row[COLMAP["college"]].strip(),
        year=int(row[COLMAP["year"]] or 1),
        self_report=scores,
        interests=interests,
        tags=[t for t in (out.get("tags") or []) if t in TAGS])
    p["name"] = row[COLMAP["name"]].strip()
    p["confidence"] = float(out["confidence"])
    p["evidence"] = evidence
    p["interest_evidence"] = kw_ev
    if dropped:
        p["dropped"] = dropped
    return p


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    path = args[0] if args else "data/responses.csv"
    rows = load_rows(path)

    if "--dry-run" in flags:
        print("=== SYSTEM ===\n" + prompts.EXTRACT_SYSTEM)
        print("=== USER (1번 응답자) ===\n" + build_prompt(rows[0]))
        print("=== SCHEMA ===\n" + json.dumps(EXTRACTION_SCHEMA,
                                              ensure_ascii=False, indent=2))
        print("\n총 %d건. 실제 호출 없음." % len(rows))
        return

    if "--one" in flags:
        rows = rows[:1]

    uids = ["u%03d" % (i + 1) for i in range(len(rows))]
    results = llm.map_call(
        list(range(len(rows))),
        lambda i: extract_one(rows[i], uids[i]))

    profiles, failed = [], []
    for uid, r in zip(uids, results):
        if isinstance(r, Exception):
            failed.append((uid, repr(r)))
            continue
        errs = validate_profile(r)
        if errs:
            failed.append((uid, errs))
            continue
        profiles.append(r)

    with open("data/profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    low = [p["user_id"] for p in profiles if p["confidence"] < 0.5]
    unsup = [(p["user_id"], p["dropped"]) for p in profiles
             if p.get("dropped")]
    nkw = [len(p["interests"]) for p in profiles]
    if any(n == 0 for n in nkw):
        sys.stderr.write("!! 관심사 0개인 프로필 있음\n")
    from collections import Counter
    tc = Counter(t for p in profiles for t in p.get("tags", []))
    sys.stderr.write("태그 분포: %r\n" % dict(tc.most_common()))
    notag = [p["user_id"] for p in profiles if not p.get("tags")]
    if notag:
        sys.stderr.write("!! 태그 0개 — 유사도 계산 불가: %r\n" % notag)
    sys.stderr.write("추출 %d/%d 성공\n" % (len(profiles), len(rows)))
    sys.stderr.write("confidence<0.5: %d건 %r\n" % (len(low), low[:10]))
    sys.stderr.write("관심사 개수 분포: %r\n"
                     % {n: nkw.count(n) for n in sorted(set(nkw))})
    if unsup:
        sys.stderr.write("근거 미달로 버린 항목 %d건:\n" % len(unsup))
        for uid, ds in unsup[:6]:
            sys.stderr.write("  %s  %s\n" % (uid, ", ".join(ds)))
    if failed:
        sys.stderr.write("실패 %d건: %r\n" % (len(failed), failed[:5]))
    sys.stderr.write("캐시: %r\n" % (llm.cache_stats(),))


if __name__ == "__main__":
    main()
