"""429 실측 — 카드 생성 병렬 호출(MAX_LLM_WORKERS)이 게이트웨이 한도에 걸리는지.

인수인계서 1순위 항목. INTEGRATION_REPORT §4 가 '미완료'로 남긴 작업이다.

왜 32명 이상인가: demo 의 8명은 그룹 2개 = 동시 요청 2개뿐이라
max_workers=8 을 전혀 포화시키지 못한다. 워커 8개를 실제로 다 쓰려면
그룹이 8개 이상, 즉 32명 이상이어야 한다.

왜 캐시를 끄는가: llm_client 는 캐시 히트 시 HTTP 요청을 보내지 않는다.
캐시된 프롬프트로는 429 가 영원히 안 뜬다.

    source .env
    python3 test_429.py 40          # 40명, 그룹 10개
    python3 test_429.py 40 4        # MAX_LLM_WORKERS=4 로 비교
"""
import os
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else None

import ai_engine.llm_client as lc          # noqa: E402
import ai_engine.matching as m             # noqa: E402
from ai_engine.demo import generate_mock_users  # noqa: E402

if WORKERS:
    m.MAX_LLM_WORKERS = WORKERS

# --- 캐시 우회: 프롬프트마다 고유 태그가 붙도록 그룹 id 를 유니크하게 ---
RUN = str(int(time.time()))

# --- 429 계측: _post 를 감싸서 실제 HTTP 상태코드를 센다 ---
stats = {"post": 0, "429": 0, "5xx": 0, "other": 0, "max_inflight": 0}
_inflight = [0]
_lock = __import__("threading").Lock()
_orig_post = lc._post


def counting_post(url, body, headers):
    with _lock:
        _inflight[0] += 1
        stats["max_inflight"] = max(stats["max_inflight"], _inflight[0])
        stats["post"] += 1
    try:
        return _orig_post(url, body, headers)
    except urllib.error.HTTPError as e:
        with _lock:
            if e.code == 429:
                stats["429"] += 1
            elif 500 <= e.code < 600:
                stats["5xx"] += 1
            else:
                stats["other"] += 1
        raise
    finally:
        with _lock:
            _inflight[0] -= 1


lc._post = counting_post


def expand(n):
    """8명 하드코딩 풀을 n명으로 늘린다. 태그·성격을 흔들어 조합을 다양화."""
    base = generate_mock_users(8)
    tags = ["운동", "여행", "요리", "음악", "영상", "독서",
            "게임", "기술", "학술", "창작", "봉사", "재테크"]
    out = []
    for i in range(n):
        u = dict(base[i % len(base)])
        u["name"] = "%s%02d" % (u["name"][:1], i + 1)
        u["user_id"] = "t%s_%03d" % (RUN, i)
        # 성격을 ±1 흔들어 그룹 구성이 매번 달라지게
        for ax in ("openness", "conscientiousness", "extraversion",
                   "agreeableness", "neuroticism"):
            u[ax] = max(1, min(5, u[ax] + ((i + hash(ax)) % 3 - 1)))
        t = [tags[(i * 5 + k) % len(tags)] for k in range(2)]
        u["tags"] = t
        u["interest_weights"] = {x: 1.0 for x in t}
        u["interest_tags"] = list(u.get("interest_tags", []))
        out.append(u)
    return out


if __name__ == "__main__":
    if os.environ.get("LLM_STUB") == "1":
        sys.exit("LLM_STUB=1 이면 HTTP 요청이 안 나간다. 끄고 실행할 것.")
    if os.environ.get("LLM_CACHE_ONLY") == "1":
        sys.exit("LLM_CACHE_ONLY=1 이면 새 프롬프트가 에러난다. 끄고 실행할 것.")

    users = expand(N)
    w = WORKERS or m.MAX_LLM_WORKERS
    print("=" * 62)
    print("429 실측  인원 %d  예상 그룹 %d  MAX_LLM_WORKERS=%d"
          % (N, N // 4, w))
    print("모델 %s  캐시 %d개 (새 프롬프트라 히트 안 됨)"
          % (lc.MODEL if hasattr(lc, "MODEL") else "?",
             lc.cache_stats().get("entries", 0)))
    print("=" * 62)

    t0 = time.time()
    try:
        cards = m.run_matching_pipeline(users)
    except Exception as e:
        print("파이프라인 예외: %r" % (e,))
        cards = []
    el = time.time() - t0

    ok = sum(1 for c in cards if c and c.get("match_reason"))
    print("\n--- 결과 ---")
    print("소요        : %.1f초" % el)
    print("카드        : %d개 생성 / %d개 성공" % (len(cards), ok))
    print("HTTP 요청   : %d회" % stats["post"])
    print("최대 동시   : %d" % stats["max_inflight"])
    print("429         : %d회" % stats["429"])
    print("5xx         : %d회" % stats["5xx"])
    print("기타 4xx    : %d회" % stats["other"])

    print("\n--- 판정 ---")
    if stats["max_inflight"] < w:
        print("! 동시 요청이 %d까지밖에 안 올라갔다 (워커 %d). 인원을 더 늘려라."
              % (stats["max_inflight"], w))
    if stats["429"] == 0:
        print("OK: 429 없음. MAX_LLM_WORKERS=%d 유지 가능." % w)
    else:
        print("429 %d회 발생 → ai_engine/matching.py 의 MAX_LLM_WORKERS 를 4로 낮출 것."
              % stats["429"])
        print("  (재시도로 성공했더라도 지연이 누적되므로 낮추는 게 맞다)")
