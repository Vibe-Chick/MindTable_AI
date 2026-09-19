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


def _weights(p):
    return p.get("interest_weights") or {k: 1.0 for k in p.get("interests", [])}


def from_api(profiles, model=None):
    """키워드를 개별 임베딩한 뒤 가중 평균으로 프로필 벡터를 만든다.

    한 문자열로 합쳐 임베딩하면 가중치를 반영할 수 없다. 리뷰로 학습한
    가중이 유사도에 실제로 반영되려면 키워드 단위여야 한다.
    고유 키워드만 임베딩하므로 호출량은 오히려 준다.
    """
    import json
    import os
    import urllib.request
    key = os.environ.get("LLM_API_KEY", "")
    model = model or os.environ.get("EMBED_MODEL", "")
    base = (os.environ.get("LLM_BASE_URL")
            or "https://api.openai.com/v1").rstrip("/")
    if not key:
        raise RuntimeError("LLM_API_KEY 미설정")
    if not model:
        raise RuntimeError("EMBED_MODEL 미설정 — 폴백으로 진행")
    vocab = sorted({k for p in profiles for k in _weights(p)})
    if not vocab:
        return profiles
    req = urllib.request.Request(
        base + "/embeddings",
        data=json.dumps({"model": model, "input": vocab}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    emb = {vocab[d["index"]]: d["embedding"]
           for d in sorted(data["data"], key=lambda d: d["index"])}
    dim = len(next(iter(emb.values())))
    for p in profiles:
        w = _weights(p)
        tot = sum(w.values()) or 1.0
        vec = [0.0] * dim
        for k, wt in w.items():
            e = emb.get(k)
            if e:
                for i in range(dim):
                    vec[i] += e[i] * wt / tot
        p["interest_vec"] = vec
    return profiles


def compose(profiles, emb_cache):
    """캐시된 키워드 임베딩으로 프로필 벡터만 다시 합성 (API 호출 없음).

    회차마다 가중치가 바뀌므로 유사도를 다시 계산해야 하는데, 키워드 집합은
    거의 안 변하니 임베딩을 재사용한다.
    """
    if not emb_cache:
        return profiles
    dim = len(next(iter(emb_cache.values())))
    for p in profiles:
        w = _weights(p)
        tot = sum(w.values()) or 1.0
        vec = [0.0] * dim
        for k, wt in w.items():
            e = emb_cache.get(k)
            if e:
                for i in range(dim):
                    vec[i] += e[i] * wt / tot
        p["interest_vec"] = vec
    return profiles


def jaccard(p, q):
    """가중 자카드. 가중치가 있으면 반영한다."""
    wp, wq = _weights(p), _weights(q)
    keys = set(wp) | set(wq)
    if not keys:
        return 0.0
    num = sum(min(wp.get(k, 0.0), wq.get(k, 0.0)) for k in keys)
    den = sum(max(wp.get(k, 0.0), wq.get(k, 0.0)) for k in keys)
    return num / den if den else 0.0


def similarity_matrix(profiles, mode="auto"):
    """태그 가중 벡터 기반. 임베딩이 있으면 키워드 임베딩을 가산한다.

    태그 공간(12차원)은 밀집해 있어서 임베딩 없이도 유사도가 붕괴하지 않는다.
    게이트웨이에 임베딩 모델이 없는 환경을 전제로 한 설계다.
    """
    base = _tag_similarity(profiles)
    if mode == "tags" or not profiles[0].get("interest_vec"):
        return base
    n = len(profiles)
    for i in range(n):
        for j in range(i + 1, n):
            e = cosine(profiles[i]["interest_vec"], profiles[j]["interest_vec"])
            v = 0.6 * base[i][j] + 0.4 * max(0.0, e)
            base[i][j] = base[j][i] = v
    return base


def tag_vector(p):
    """태그 가중 벡터. 유사도는 전적으로 이 공간에서 계산한다."""
    from schema import TAGS
    w = _weights(p)
    return [w.get(t, 0.0) for t in TAGS]


def _tag_similarity(profiles):
    n = len(profiles)
    vs = [tag_vector(p) for p in profiles]
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
        for j in range(i + 1, n):
            # 태그 코사인이 주(主), 자유 키워드 일치는 소폭 가산
            v = 0.85 * cosine(vs[i], vs[j]) + 0.15 * jaccard(profiles[i],
                                                             profiles[j])
            m[i][j] = m[j][i] = max(0.0, min(1.0, v))
    return m
