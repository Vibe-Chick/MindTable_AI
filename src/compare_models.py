"""모델 선택을 감이 아니라 측정으로 한다.

같은 응답을 여러 모델로 채점하고, 팀원 수기 채점과의 일치도를 비교한다.
전체 비용이 $1 수준이라 안 할 이유가 없고, 질의응답에서
"모델은 어떻게 골랐나요"에 답이 생긴다.

준비:
  data/responses.csv        실제 응답 (최소 20건)
  data/manual_scores.csv    팀원 수기 채점: user_id,O,C,E,A,N

사용:
  source .env
  python3 src/compare_models.py gpt-5.6-luna gpt-5.6-terra gpt-5.6-sol
"""
import csv
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import llm                                                    # noqa: E402
import prompts                                                # noqa: E402
from schema import AXES, EXTRACTION_SCHEMA, LONG_TO_AXIS      # noqa: E402

import extract as ex                                          # noqa: E402


def load_manual(path="data/manual_scores.csv"):
    if not os.path.exists(path):
        sys.exit("수기 채점 파일이 없다: %s\n"
                 "형식: user_id,O,C,E,A,N (팀원 2명이 20건 채점)" % path)
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["user_id"]] = {a: int(row[a]) for a in AXES if row.get(a)}
    return out


def score_with(model, rows, uids):
    old = llm.MODEL
    llm.MODEL = model
    got = {}
    for row, uid in zip(rows, uids):
        try:
            out = llm.call(ex.build_prompt(row), EXTRACTION_SCHEMA,
                           system=prompts.EXTRACT_SYSTEM, temp=0.0,
                           tag="cmp:%s:%s" % (model, uid))
            got[uid] = {ax: int(out[lg]) for lg, ax in LONG_TO_AXIS.items()}
        except Exception as e:
            sys.stderr.write("  %s %s 실패: %r\n" % (model, uid, e))
    llm.MODEL = old
    return got


def agreement(manual, got):
    res = {}
    for a in AXES:
        vs = [(manual[u][a], got[u][a]) for u in got
              if u in manual and a in manual[u]]
        if len(vs) < 3:
            res[a] = None
            continue
        n = len(vs)
        mh = sum(h for h, _ in vs) / n
        ml = sum(l for _, l in vs) / n
        cov = sum((h - mh) * (l - ml) for h, l in vs)
        vh = sum((h - mh) ** 2 for h, _ in vs) ** 0.5
        vl = sum((l - ml) ** 2 for _, l in vs) ** 0.5
        res[a] = {
            "r": (cov / (vh * vl)) if vh and vl else 0.0,
            "within1": sum(1 for h, l in vs if abs(h - l) <= 1) / n,
            "n": n,
        }
    return res


if __name__ == "__main__":
    models = sys.argv[1:] or [llm.MODEL]
    manual = load_manual()
    rows = ex.load_rows("data/responses.csv")
    uids = ["u%03d" % (i + 1) for i in range(len(rows))]
    keep = [i for i, u in enumerate(uids) if u in manual]
    rows = [rows[i] for i in keep]
    uids = [uids[i] for i in keep]
    print("수기 채점된 %d건으로 비교\n" % len(uids))

    table = {}
    for m in models:
        print("채점 중: %s" % m)
        table[m] = agreement(manual, score_with(m, rows, uids))

    print("\n상관계수 r (사람 채점 대비)")
    print("%-18s %s   평균" % ("model", "  ".join("%5s" % a for a in AXES)))
    best = None
    for m in models:
        rs = [table[m][a]["r"] for a in AXES if table[m][a]]
        avg = sum(rs) / len(rs) if rs else 0
        print("%-18s %s  %5.3f"
              % (m, "  ".join("%5.2f" % table[m][a]["r"] if table[m][a]
                              else "    -" for a in AXES), avg))
        if best is None or avg > best[1]:
            best = (m, avg)

    print("\n±1점 이내 일치 비율")
    for m in models:
        ws = [table[m][a]["within1"] for a in AXES if table[m][a]]
        print("%-18s %s  %5.3f"
              % (m, "  ".join("%5.2f" % table[m][a]["within1"] if table[m][a]
                              else "    -" for a in AXES),
                 sum(ws) / len(ws) if ws else 0))

    if best:
        print("\n권장: LLM_MODEL=%s  (평균 r=%.3f)" % best)
    print("캐시: %r" % (llm.cache_stats(),))
