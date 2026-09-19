"""전체 파이프라인. 이 한 줄이 out/ 을 처음부터 다시 만든다.

    LLM_STUB=1 python3 run.py --mock 60 --rounds 3   # API 없이 전 구간
    python3 run.py --rounds 3                        # 실제 (키 필요)

out/ 이 프론트엔드와의 계약이다. D는 이 파일들만 읽는다.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import cards                                                  # noqa: E402
import embed                                                  # noqa: E402
import llm                                                    # noqa: E402
import match                                                  # noqa: E402
import metrics                                                # noqa: E402
import mock                                                   # noqa: E402
import simulate                                               # noqa: E402
from schema import effective                                  # noqa: E402

OUT = "out"


def dump(name, data):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("  out/%s" % name)


def profile_view(profiles):
    """프론트엔드용. 숨은 정답(_true 등)은 내보내지 않는다."""
    rows = []
    for p in profiles:
        eff = effective(p)
        rows.append({
            "user_id": p["user_id"], "name": p.get("name", ""),
            "school": p["school"], "major": p["major"],
            "college": p["college"], "year": p["year"],
            "interests": p["interests"],
            "self_report": p["self_report"],
            "effective": {a: round(v, 2) for a, v in eff.items()},
            "beta": round(p["beta"], 3),
            "confidence": p.get("confidence", 0.0),
        })
    return rows


def group_view(groups, profiles):
    return [[profiles[i]["user_id"] for i in g] for g in groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", type=int, default=0,
                    help="합성 프로필 N건으로 실행")
    ap.add_argument("--profiles", default="data/profiles.json")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    print("=" * 62)
    if a.mock:
        profiles = mock.gen(a.mock)
        print("mock 프로필 %d건" % len(profiles))
    else:
        profiles = json.load(open(a.profiles, encoding="utf-8"))
        print("프로필 %d건 (%s)" % (len(profiles), a.profiles))

    # 1) 유사도 -------------------------------------------------------
    if not a.mock and not profiles[0].get("interest_vec"):
        try:
            embed.from_api(profiles)
            print("임베딩 API 완료")
        except Exception as e:
            sys.stderr.write("[run] 임베딩 실패 → 폴백: %r\n" % e)
    sim = embed.similarity_matrix(profiles)

    # 2) 편성 벤치마크 ------------------------------------------------
    bench = match.benchmark(profiles, sim)
    o = bench["objective"]
    print("편성  random=%.3f  greedy=%.3f  local=%.3f"
          % (o["random"], o["greedy"], o["local_search"]))

    # 3) 그룹 카드 ----------------------------------------------------
    cs = cards.build_all(bench["groups"], profiles, 1)
    print("카드 %d장 (LLM 실패 %d장)"
          % (len(cs), sum(1 for c in cs if not c.get("reason"))))

    # 4) 회차 루프 ----------------------------------------------------
    has_truth = "_true" in profiles[0]
    if has_truth:
        hist, snaps, final = simulate.run(profiles, rounds=a.rounds)
        print("시뮬레이션 %d회차: 틀린집단 오차 %.4f → %.4f"
              % (a.rounds, hist[0]["err_mismatch"], hist[-1]["err_mismatch"]))
    else:
        hist, snaps, final = [], [profiles], profiles
        print("숨은 정답 없음 → 회차 루프 생략 (실측 리뷰는 review.py 로)")

    # 5) 출력 ---------------------------------------------------------
    print("\n출력:")
    dump("groups_r1.json", {"groups": group_view(bench["groups"], profiles),
                            "cards": cs})
    dump("baseline_r1.json",
         {"groups": group_view(bench["groups_random"], profiles),
          "objective": o["random"]})
    for i, snap in enumerate(snaps, start=1):
        dump("profiles_r%d.json" % i, profile_view(snap))
    if not snaps:
        dump("profiles_r1.json", profile_view(profiles))
    dump("profiles_final.json", profile_view(final))
    m = metrics.collect(hist, bench, final, cs,
                        manual_path="data/manual_scores.csv")
    metrics.write(os.path.join(OUT, "metrics.json"), m)
    print("  out/metrics.json")

    print("\n캐시: %r" % (llm.cache_stats(),))
    if llm.STUB:
        print("\n*** LLM_STUB=1 로 실행됨 — reason/icebreakers 는 가짜입니다 ***")
    print("=" * 62)


if __name__ == "__main__":
    main()
