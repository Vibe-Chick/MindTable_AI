"""LLM 호출 래퍼 — 캐시 + 재시도 + 동시성 제한.

이 프로젝트에서 가장 중요한 40줄. 캐시가 있으면 재실행이 공짜·즉시가 되고,
최적화기를 50번 튜닝해도 API를 건드리지 않는다.

의존성 없음 (stdlib만). pip install 불필요.

환경변수:
    LLM_PROVIDER  openai | anthropic | gemini   (기본 openai)
    LLM_MODEL     모델 ID
    LLM_API_KEY   API 키
    LLM_BASE_URL  게이트웨이 주소. OpenAI 호환 프록시를 쓸 때 지정한다.
                  예) 국민대: https://ai.cs.kookmin.ac.kr/v1
    LLM_CACHE_ONLY=1   캐시에 없으면 에러. 무대 시연 때 반드시 켠다.
    LLM_STUB=1         API 없이 스키마에 맞는 가짜 응답 생성. 개발 전용.
                       결과를 캐시에 쓰지 않으므로 실제 실행을 오염시키지 않는다.
"""
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
API_KEY = os.environ.get("LLM_API_KEY", "")
CACHE_ONLY = os.environ.get("LLM_CACHE_ONLY", "") == "1"
STUB = os.environ.get("LLM_STUB", "") == "1"
_stub_warned = False

# 모델 ID는 벤더 페이지에서 확인 후 환경변수로 덮어쓸 것
DEFAULT_MODEL = {
    "openai":    "gpt-5.6-luna",
    "anthropic": "SET_LLM_MODEL",
    "gemini":    "SET_LLM_MODEL",
}
MODEL = os.environ.get("LLM_MODEL") or DEFAULT_MODEL.get(PROVIDER, "")

DEFAULT_BASE = {
    "openai":    "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini":    "https://generativelanguage.googleapis.com/v1beta",
}
BASE_URL = (os.environ.get("LLM_BASE_URL")
            or DEFAULT_BASE.get(PROVIDER, "")).rstrip("/")

MAX_RETRY = 3
WORKERS = 4          # 신규 키는 동시 요청이 많으면 429. 늘리지 말 것


class LLMError(RuntimeError):
    pass


def _cache_key(payload):
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _cache_path(key):
    return os.path.join(CACHE_DIR, key + ".json")


def _post(url, body, headers):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def _build(prompt, schema, system, temp):
    """벤더별 요청 본문 + 응답 파서를 함께 돌려준다."""
    if PROVIDER == "openai":
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body = {
            "model": MODEL, "messages": msgs, "temperature": temp,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True,
                                "schema": schema},
            },
        }
        return (BASE_URL + "/chat/completions", body,
                {"Authorization": "Bearer " + API_KEY},
                lambda r: json.loads(r["choices"][0]["message"]["content"]))

    if PROVIDER == "anthropic":
        body = {
            "model": MODEL, "max_tokens": 1024, "temperature": temp,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"name": "emit", "description": "결과 출력",
                       "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "emit"},
        }
        if system:
            body["system"] = system
        return (BASE_URL + "/messages", body,
                {"x-api-key": API_KEY, "anthropic-version": "2023-06-01"},
                lambda r: next(b["input"] for b in r["content"]
                               if b.get("type") == "tool_use"))

    if PROVIDER == "gemini":
        # 주의: Gemini responseSchema는 OpenAPI 서브셋만 받는다.
        # additionalProperties / enum 조합에서 거부되면 스키마를 단순화할 것.
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temp,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = BASE_URL + "/models/" + MODEL + ":generateContent?key=" + API_KEY
        return (url, body, {},
                lambda r: json.loads(
                    r["candidates"][0]["content"]["parts"][0]["text"]))

    raise LLMError("unknown LLM_PROVIDER: " + PROVIDER)


_STUB_WORDS = ["등산", "홈서버", "재즈", "베이킹", "독립영화", "보드게임",
               "배낭여행", "클라이밍", "사진", "철학"]


def _stub_value(spec, seed):
    """스키마를 만족하는 결정적 가짜 값. 개발용."""
    rnd = random.Random(seed)
    t = spec.get("type")
    if "enum" in spec:
        return rnd.choice(spec["enum"])
    if t == "integer":
        return rnd.randint(spec.get("minimum", 1), spec.get("maximum", 5))
    if t == "number":
        lo, hi = spec.get("minimum", 0.0), spec.get("maximum", 1.0)
        return round(rnd.uniform(lo, hi), 2)
    if t == "boolean":
        return rnd.random() < 0.5
    if t == "array":
        n = spec.get("minItems", 3)
        item = spec.get("items", {})
        if item.get("type") == "string":
            return rnd.sample(_STUB_WORDS, min(n, len(_STUB_WORDS)))
        return [_stub_value(item, seed + i) for i in range(n)]
    return "(stub)"


def _stub(prompt, schema, seed):
    global _stub_warned
    if not _stub_warned:
        sys.stderr.write(
            "\n*** LLM_STUB=1 — 가짜 응답입니다. 결과 수치를 신뢰하지 마세요. ***\n\n")
        _stub_warned = True
    props = schema.get("properties", {})
    return {k: _stub_value(v, seed + i)
            for i, (k, v) in enumerate(sorted(props.items()))}


def call(prompt, schema, system=None, temp=0.0, tag=""):
    """구조화 출력 1회. 캐시 히트면 API를 건드리지 않는다."""
    key = _cache_key({"p": prompt, "s": schema, "sys": system,
                      "t": temp, "m": MODEL, "v": PROVIDER})
    if STUB:
        # 캐시에 쓰지 않는다. 가짜 응답이 실제 실행에 섞이면 안 된다.
        return _stub(prompt, schema, int(key[:8], 16))

    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["result"]

    if CACHE_ONLY:
        raise LLMError(
            "LLM_CACHE_ONLY=1 인데 캐시에 없음 (tag=%s, key=%s). "
            "무대 시연 전에 전체 파이프라인을 한 번 돌려 캐시를 채울 것." % (tag, key))
    if not API_KEY:
        raise LLMError("LLM_API_KEY 미설정")
    if MODEL.startswith("SET_"):
        raise LLMError("LLM_MODEL 미설정 (provider=%s)" % PROVIDER)

    url, body, headers, parse = _build(prompt, schema, system, temp)
    last = None
    for attempt in range(MAX_RETRY):
        try:
            result = parse(_post(url, body, headers))
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"tag": tag, "model": MODEL, "prompt": prompt,
                           "result": result}, f, ensure_ascii=False, indent=2)
            return result
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529):
                time.sleep((2 ** attempt) + random.random())
                continue
            raise LLMError("HTTP %s: %s" % (e.code, e.read()[:400])) from e
        except Exception as e:                      # 파싱 실패 등
            last = e
            time.sleep(1 + attempt)
    raise LLMError("재시도 %d회 실패 (tag=%s): %r" % (MAX_RETRY, tag, last))


def map_call(items, fn, workers=WORKERS):
    """동시성 제한 배치 실행. 실패는 예외 객체로 담아 돌려준다."""
    out = [None] * len(items)

    def run(i):
        try:
            out[i] = fn(items[i])
        except Exception as e:
            out[i] = e

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run, range(len(items))))
    return out


def list_models():
    """게이트웨이가 실제로 내주는 모델 목록. 권한 그룹에 따라 달라진다."""
    req = urllib.request.Request(
        BASE_URL + "/models",
        headers={"Authorization": "Bearer " + API_KEY, "x-api-key": API_KEY})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    return sorted(m.get("id", "?") for m in data.get("data", []))


def cache_stats():
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    return {"entries": len(files), "dir": CACHE_DIR}
