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
from schema import (EXTRACTION_SCHEMA, LONG_TO_AXIS,          # noqa: E402
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
    p = new_profile(
        user_id=uid,
        school=row[COLMAP["school"]].strip(),
        major=row[COLMAP["major"]].strip(),
        college=row[COLMAP["college"]].strip(),
        year=int(row[COLMAP["year"]] or 1),
        self_report={ax: int(out[long]) for long, ax in LONG_TO_AXIS.items()},
        interests=out["interests"])
    p["name"] = row[COLMAP["name"]].strip()
    p["confidence"] = float(out["confidence"])
    p["evidence"] = out["evidence"]
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
    sys.stderr.write("추출 %d/%d 성공\n" % (len(profiles), len(rows)))
    sys.stderr.write("confidence<0.5: %d건 %r\n" % (len(low), low[:10]))
    if failed:
        sys.stderr.write("실패 %d건: %r\n" % (len(failed), failed[:5]))
    sys.stderr.write("캐시: %r\n" % (llm.cache_stats(),))


if __name__ == "__main__":
    main()
