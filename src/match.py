"""편성 최적화기 — 이 프로젝트에서 LLM이 아닌 부분.

N명을 GROUP_SIZE 인 그룹으로 분할하며 총 score를 최대화한다 (NP-hard 집합 분할).
최적해는 필요 없다. 랜덤보다 낫다는 것을 숫자로 보이면 된다.

    python3 src/match.py data/profiles_mock.json
"""
import json
import random
import statistics as st
import sys
from itertools import combinations

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from schema import (ALPHA, DELTA_REPEAT, EPSILON, GAMMA, GROUP_SIZE,  # noqa
                    LOCAL_SEARCH_MAX_ITER, effective)
import embed  # noqa: E402


# --- 목적함수 ------------------------------------------------------------

def complement(members):
    """성격 보완도.

    외향성은 분산을 원한다 (전원 조용 = 자리가 죽고, 전원 시끄러움 = 산만).
    개방성·우호성은 유사를 원한다.
    """
    if len(members) < 2:
        return 0.0
    eff = [effective(p) for p in members]
    e_std = st.pstdev([e["E"] for e in eff])
    e_term = 1.0 - abs(e_std - 1.0) / 2.0          # 목표 분산 ≈ 1.0
    oa = (st.pstdev([e["O"] for e in eff]) +
          st.pstdev([e["A"] for e in eff])) / 2.0
    return e_term - oa / 2.0


def diversity(members):
    n = len(members)
    maj = len({p["major"] for p in members}) / n
    col = len({p["college"] for p in members}) / n
    sch = len({p["school"] for p in members}) / n
    return 0.5 * maj + 0.3 * col + 0.2 * sch


def score_group(idx, profiles, sim):
    members = [profiles[i] for i in idx]
    if len(idx) < 2:
        return 0.0
    pairs = list(combinations(idx, 2))
    interest = sum(sim[i][j] for i, j in pairs) / len(pairs)
    beta = sum(p["beta"] for p in members) / len(members)
    repeat = sum(1 for i, j in pairs
                 if profiles[j]["user_id"] in profiles[i].get("met", []))
    return (ALPHA * interest
            + beta * diversity(members)
            + GAMMA * complement(members)
            - DELTA_REPEAT * repeat)


def total(groups, profiles, sim):
    return sum(score_group(g, profiles, sim) for g in groups)


# --- 전략 ----------------------------------------------------------------

def random_partition(profiles, sim, seed=0):
    """baseline. 이 숫자가 없으면 나머지 숫자가 의미 없다."""
    idx = list(range(len(profiles)))
    random.Random(seed).shuffle(idx)
    groups = [idx[i:i + GROUP_SIZE] for i in range(0, len(idx), GROUP_SIZE)]
    full = [g for g in groups if len(g) == GROUP_SIZE]
    leftover = [i for g in groups if len(g) < GROUP_SIZE for i in g]
    return full, leftover


def greedy(profiles, sim, explore=True, seed=0):
    """시드 + 한계이득 최대 충원. 그룹당 1자리는 탐색용으로 예약한다."""
    rng = random.Random(seed)
    pool = set(range(len(profiles)))
    groups = []

    while len(pool) >= GROUP_SIZE:
        seed_i = max(pool, key=lambda i: sum(sim[i][j] for j in pool))
        g = [seed_i]
        pool.discard(seed_i)

        # 탐색 슬롯: 시드와 유사도 하위권에서 1명을 의도적으로 넣는다.
        # 최적화 후 깨는 방식(post-hoc swap)은 목적함수 값을 망가뜨리므로 금지.
        n_normal = GROUP_SIZE - 1 - (1 if explore else 0)

        for _ in range(n_normal):
            if not pool:
                break
            best = max(pool, key=lambda c: score_group(g + [c], profiles, sim))
            g.append(best)
            pool.discard(best)

        if explore and pool:
            ranked = sorted(pool, key=lambda c: sim[seed_i][c])
            k = max(1, int(len(ranked) * EPSILON))
            pick = rng.choice(ranked[:k])          # 유사도 하위 25%에서
            g.append(pick)
            pool.discard(pick)

        groups.append(g)

    return groups, sorted(pool)     # 남는 인원 = 다음 라운드 이월


def local_search(groups, profiles, sim, max_iter=LOCAL_SEARCH_MAX_ITER):
    """두 그룹 간 멤버 스왑. 총점이 오르면 수용."""
    groups = [list(g) for g in groups]
    cur = total(groups, profiles, sim)
    it = 0
    improved = True
    while improved and it < max_iter:
        improved = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                for x in range(len(groups[a])):
                    for y in range(len(groups[b])):
                        it += 1
                        if it > max_iter:
                            return groups, cur, it
                        groups[a][x], groups[b][y] = groups[b][y], groups[a][x]
                        new = total(groups, profiles, sim)
                        if new > cur + 1e-9:
                            cur = new
                            improved = True
                        else:
                            groups[a][x], groups[b][y] = (groups[b][y],
                                                          groups[a][x])
    return groups, cur, it


# --- 실행 ----------------------------------------------------------------

def benchmark(profiles, sim):
    rnd, rnd_left = random_partition(profiles, sim)
    grd, grd_left = greedy(profiles, sim, explore=True)
    opt, opt_score, iters = local_search(grd, profiles, sim)
    return {
        "n_users": len(profiles),
        "n_groups": len(opt),
        "leftover": len(grd_left),
        "objective": {
            "random": round(total(rnd, profiles, sim), 4),
            "greedy": round(total(grd, profiles, sim), 4),
            "local_search": round(opt_score, 4),
        },
        "local_search_iters": iters,
        "groups": opt,
        "groups_random": rnd,
        "carry_over": grd_left,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/profiles_mock.json"
    profiles = json.load(open(path, encoding="utf-8"))
    sim = embed.similarity_matrix(profiles)
    r = benchmark(profiles, sim)

    o = r["objective"]
    print("users %d → groups %d (이월 %d명)"
          % (r["n_users"], r["n_groups"], r["leftover"]))
    print("objective  random=%.3f  greedy=%.3f  local=%.3f  (iters=%d)"
          % (o["random"], o["greedy"], o["local_search"],
             r["local_search_iters"]))
    gain = (o["local_search"] - o["random"]) / abs(o["random"]) * 100
    print("random 대비 향상: %+.1f%%" % gain)
    assert o["local_search"] >= o["greedy"] >= o["random"], "순서 위반"
    print("OK: local >= greedy >= random")

    g = r["groups"][0]
    print("\n샘플 그룹:")
    for i in g:
        p = profiles[i]
        print("  %s %s / %s %s / %s"
              % (p["user_id"], p.get("name", ""), p["school"], p["major"],
                 ", ".join(p["interests"])))
