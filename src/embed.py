"""관심사 임베딩 → 유사도 행렬.

세 가지 경로:
  from_api      실제 임베딩 API (운영 경로)
  jaccard       API 실패 시 폴백. 파이프라인이 멈추는 것보단 낫다
  synthetic     mock 데이터 전용. 테마 라벨로 유사도를 합성한다

※ synthetic은 mock.py가 만든 'theme' 필드에 의존한다. 실제 응답에는 그 필드가
  없으므로 운영 경로에서는 절대 호출되지 않는다. 혼동 방지를 위해 경고를 찍는다.
"""
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def from_api(profiles, model=None):
    """관심사 키워드 3개를 한 문자열로 합쳐 임베딩. interest_vec에 기록."""
    import json
    import os
    import urllib.request
    key = os.environ.get("LLM_API_KEY", "")
    model = model or os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    if not key:
        raise RuntimeError("LLM_API_KEY 미설정")
    texts = [", ".join(p["interests"]) for p in profiles]
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    for p, item in zip(profiles, sorted(data["data"], key=lambda d: d["index"])):
        p["interest_vec"] = item["embedding"]
    return profiles


def jaccard(p, q):
    a, b = set(p["interests"]), set(q["interests"])
    return len(a & b) / len(a | b) if (a | b) else 0.0


def similarity_matrix(profiles, mode="auto"):
    n = len(profiles)
    if mode == "synthetic" or (mode == "auto" and "theme" in profiles[0]
                               and not profiles[0].get("interest_vec")):
        sys.stderr.write(
            "[embed] WARNING: synthetic similarity (mock 전용). "
            "실제 데이터에는 from_api를 쓸 것.\n")
        return _synthetic(profiles)
    if profiles[0].get("interest_vec"):
        return [[cosine(profiles[i]["interest_vec"], profiles[j]["interest_vec"])
                 if i != j else 1.0 for j in range(n)] for i in range(n)]
    sys.stderr.write("[embed] WARNING: jaccard 폴백 사용\n")
    return [[jaccard(profiles[i], profiles[j]) if i != j else 1.0
             for j in range(n)] for i in range(n)]


def _synthetic(profiles):
    import random
    rng = random.Random(7)
    n = len(profiles)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
        for j in range(i + 1, n):
            pi, pj = profiles[i], profiles[j]
            shared = set(pi["interests"]) & set(pj["interests"])
            if pi["theme"] == pj["theme"]:
                base = 0.75
            elif shared:
                base = 0.50
            else:
                base = 0.12
            v = max(0.0, min(1.0, base + rng.gauss(0, 0.07)))
            m[i][j] = m[j][i] = v
    return m
