"""Stage 7 — 보정 루프.

이 프로젝트의 차별점 그 자체. 리뷰에서 뽑은 델타로 '본인' 프로필을 갱신한다.
상대방을 평가하지 않는다. 갱신 대상은 항상 리뷰를 쓴 사람 자신이다.

델타 소스는 두 가지이고 적용 로직(apply_delta)은 공유한다:
  - 실측: 리뷰 원문 → LLM → 델타            (extract_delta)
  - 시뮬: 숨은 정답 → 합성 델타              (simulate.py)
"""
import json
import sys
from itertools import combinations

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import llm                                                   # noqa: E402
import prompts                                               # noqa: E402
from schema import (BEHAVIOR_CLIP, BEHAVIOR_DECAY, BETA_MAX,  # noqa
                    BETA_MIN, BETA_STEP, DELTA_SCHEMA,
                    DIVERGENCE_THRESHOLD, INTEREST_DECAY, INTEREST_DOWN,
                    INTEREST_DROP, INTEREST_MAX, INTEREST_MIN,
                    INTEREST_NEW, INTEREST_UP, LONG_TO_AXIS, LR,
                    clamp, effective, max_divergence)

Q4_MORE_SIMILAR = "더 비슷"
Q4_SAME = "지금 정도"
Q4_MORE_DIFFERENT = "더 달라도 됨"


def extract_delta(profile, review):
    """리뷰 원문 → LLM → 축별 델타. 실측 경로."""
    eff = effective(profile)
    prompt = prompts.REVIEW_USER.format(
        O=round(eff["O"], 1), C=round(eff["C"], 1), E=round(eff["E"], 1),
        A=round(eff["A"], 1), N=round(eff["N"], 1),
        r1=review.get("r1", "").strip() or "(무응답)",
        r2=review.get("r2", "").strip() or "(무응답)",
        r3=review.get("r3", "").strip() or "(무응답)")
    return llm.call(prompt, DELTA_SCHEMA, system=prompts.REVIEW_SYSTEM,
                    temp=0.0, tag="review:" + profile["user_id"])


def apply_delta(p, delta, q4=None):
    """델타 적용 — 지수이동평균 갱신.

        behavior <- behavior*(1-DECAY) + LR*confidence*delta

    감쇠항이 핵심이다. 단순 누적(behavior += LR*delta)으로 하면 리뷰 잡음이
    랜덤워크로 쌓여서, 자기보고가 이미 정확했던 사람의 프로필까지 망가진다.
    감쇠가 있으면 일관된 신호만 살아남고 1회성 잡음은 사라진다.

    confidence 가중도 같은 이유다. "재밌었어요"뿐인 후기는 프로필을 거의
    못 움직인다.

    자기보고(self_report)는 절대 건드리지 않는다 — 발표 시각화의 기준선이다.
    """
    conf = clamp(float(delta.get("confidence", 1.0) or 0.0), 0.0, 1.0)
    for long, ax in LONG_TO_AXIS.items():
        d = float(delta.get(long, 0.0) or 0.0)
        b = p["behavior"][ax] * (1.0 - BEHAVIOR_DECAY) + LR * conf * d
        p["behavior"][ax] = clamp(b, -BEHAVIOR_CLIP, BEHAVIOR_CLIP)
    apply_topics(p, delta.get("worked_topics") or [],
                 delta.get("dead_topics") or [], conf)

    if q4 == Q4_MORE_SIMILAR:
        p["beta"] = clamp(p["beta"] - BETA_STEP, BETA_MIN, BETA_MAX)
    elif q4 == Q4_MORE_DIFFERENT:
        p["beta"] = clamp(p["beta"] + BETA_STEP, BETA_MIN, BETA_MAX)
    return p


def apply_topics(p, worked, dead, conf=1.0):
    """실제로 통한 주제 / 죽은 주제로 관심사 가중을 갱신한다.

    설문에 쓴 관심사는 '자기가 생각하는 나'이고, 여기서 들어오는 신호는
    '실제 자리에서 일어난 일'이다. 이 둘이 갈리는 게 자기보고 편향의
    가장 눈에 보이는 형태다.

    가중치는 1.0 쪽으로 천천히 수축한다(INTEREST_DECAY). 없으면 한 번
    언급된 주제가 영구히 프로필을 지배한다.
    """
    w = p.setdefault("interest_weights",
                     {k: 1.0 for k in p.get("interests", [])})

    for k in list(w):                       # 1.0 쪽으로 수축
        w[k] += (1.0 - w[k]) * INTEREST_DECAY

    for t in worked:
        t = t.strip()
        if not t:
            continue
        cur = w.get(t, INTEREST_NEW)
        w[t] = clamp(cur + INTEREST_UP * conf, INTEREST_MIN, INTEREST_MAX)

    for t in dead:
        t = t.strip()
        if t in w:
            w[t] = clamp(w[t] - INTEREST_DOWN * conf,
                         INTEREST_MIN, INTEREST_MAX)

    for k in [k for k, v in w.items() if v < INTEREST_DROP]:
        del w[k]

    # interests 는 표시용. 가중 상위 3개를 유지한다.
    p["interests"] = [k for k, _ in sorted(w.items(), key=lambda kv: -kv[1])][:3]
    return p


def record_meeting(profiles, group_idx, group_id):
    """같은 자리에 앉은 쌍을 기록. 다음 회차 반복 매칭 페널티로 쓰인다."""
    for i in group_idx:
        profiles[i]["history"].append(group_id)
    for i, j in combinations(group_idx, 2):
        profiles[i]["met"].append(profiles[j]["user_id"])
        profiles[j]["met"].append(profiles[i]["user_id"])


def skip_no_review(p):
    """미제출 처리. 델타 0 — 루프가 멈추는 것보단 낫다."""
    return p


def divergence_report(profiles):
    """자기보고 vs 보정 후. 발표 5번 장면의 재료."""
    rows = []
    for p in profiles:
        ax, d = max_divergence(p)
        if abs(d) >= DIVERGENCE_THRESHOLD:
            rows.append({
                "user_id": p["user_id"], "name": p.get("name", ""),
                "axis": ax,
                "self_report": p["self_report"][ax],
                "effective": round(effective(p)[ax], 2),
                "delta": round(d, 2),
            })
    return sorted(rows, key=lambda r: -abs(r["delta"]))


def main():
    """실측 경로: reviews_r{n}.json → LLM → profiles 갱신."""
    rnd = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    profiles = json.load(open("data/profiles.json", encoding="utf-8"))
    reviews = json.load(open("data/reviews_r%d.json" % rnd, encoding="utf-8"))
    by_id = {p["user_id"]: p for p in profiles}

    submitted = missing = 0
    for uid, rv in reviews.items():
        p = by_id.get(uid)
        if p is None:
            continue
        if not any(rv.get(k, "").strip() for k in ("r1", "r2", "r3")):
            skip_no_review(p)
            missing += 1
            continue
        try:
            d = extract_delta(p, rv)
        except Exception as e:
            sys.stderr.write("[review] %s 델타 추출 실패 → 0 처리: %r\n" % (uid, e))
            missing += 1
            continue
        apply_delta(p, d, rv.get("q4"))
        p.setdefault("delta_log", []).append(
            {"round": rnd, "delta": d, "evidence": d.get("evidence", "")})
        submitted += 1

    json.dump(profiles, open("data/profiles.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    sys.stderr.write("보정 완료: 반영 %d / 미제출·실패 %d\n" % (submitted, missing))
    for r in divergence_report(profiles)[:5]:
        sys.stderr.write("  괴리 %s %s: 자기보고 %d → %.2f (%+.2f)\n"
                         % (r["user_id"], r["axis"], r["self_report"],
                            r["effective"], r["delta"]))


if __name__ == "__main__":
    main()
