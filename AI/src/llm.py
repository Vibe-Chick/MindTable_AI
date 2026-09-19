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
    LLM_STRUCTURED     tools(기본) | json_schema | json_object
                       ※ 국민대 게이트웨이는 json_schema/json_object 를 조용히
                         무시한다(에러 없이 한국어 키로 아무 JSON이나 반환).
                         probe.py 로 확인됨. tools 만 실제로 동작한다.
    LLM_DEBUG=1        원시 응답을 stderr 에 덤프. 게이트웨이 진단용.
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
DEBUG = os.environ.get("LLM_DEBUG", "") == "1"
STRUCTURED = os.environ.get("LLM_STRUCTURED", "tools").lower()
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


class ParseError(LLMError):
    """응답은 왔는데 형식이 다르다. 재시도해도 같은 결과이므로 즉시 중단."""


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


def _strip_fence(t):
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _json_or_die(txt, raw):
    txt = _strip_fence(txt)
    if not txt:
        raise ParseError("빈 응답. 원시: " + json.dumps(raw, ensure_ascii=False)[:600])
    try:
        return json.loads(txt)
    except Exception:
        # 앞뒤에 설명이 붙은 경우 첫 JSON 오브젝트만 뽑는다
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(txt[i:j + 1])
            except Exception:
                pass
        raise ParseError("JSON 아님: " + txt[:400])


def _parse_openai(r):
    """게이트웨이마다 결과 위치가 다르다.

    New API 가 Claude 를 중계할 때 response_format=json_schema 를 tool call 로
    변환하는 경우가 있어, 답이 content 가 아니라 tool_calls 에 들어온다.
    """
    if DEBUG:
        sys.stderr.write("[llm] raw: %s\n"
                         % json.dumps(r, ensure_ascii=False)[:1200])
    if "choices" not in r:
        raise ParseError("choices 없음. 원시: "
                         + json.dumps(r, ensure_ascii=False)[:600])
    msg = r["choices"][0].get("message", {}) or {}
    txt = msg.get("content")
    if isinstance(txt, list):                    # 일부 게이트웨이의 블록 형식
        txt = "".join(b.get("text", "") for b in txt if isinstance(b, dict))
    if not (txt or "").strip():
        for tc in (msg.get("tool_calls") or []):
            a = (tc.get("function") or {}).get("arguments")
            if a:
                txt = a
                break
    if not (txt or "").strip():
        fc = msg.get("function_call") or {}
        txt = fc.get("arguments")
    return _json_or_die(txt, r)


def _build(prompt, schema, system, temp):
    """벤더별 요청 본문 + 응답 파서를 함께 돌려준다."""
    if PROVIDER == "openai":
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body = {"model": MODEL, "messages": msgs, "temperature": temp}

        if STRUCTURED == "tools":
            # 함수 호출로 스키마를 강제한다. Claude 계열이든 오픈소스 모델이든
            # OpenAI 호환 게이트웨이에서 가장 널리 실제 동작하는 방식.
            body["tools"] = [{"type": "function",
                              "function": {"name": "emit",
                                           "description": "결과 출력",
                                           "parameters": schema}}]
            body["tool_choice"] = {"type": "function",
                                   "function": {"name": "emit"}}
        elif STRUCTURED == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True,
                                "schema": schema}}
        elif STRUCTURED == "json_object":
            body["response_format"] = {"type": "json_object"}

        return (BASE_URL + "/chat/completions", body,
                {"Authorization": "Bearer " + API_KEY}, _parse_openai)

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
                _parse_anthropic)

    if PROVIDER == "gemini":  # noqa
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
    if t == "object":
        return {k: _stub_value(v, seed + i)
                for i, (k, v) in enumerate(sorted(
                    spec.get("properties", {}).items()))}
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


def _parse_anthropic(r):
    if DEBUG:
        sys.stderr.write("[llm] raw: %s\n"
                         % json.dumps(r, ensure_ascii=False)[:1200])
    for b in r.get("content", []):
        if b.get("type") == "tool_use":
            return b["input"]
    txt = "".join(b.get("text", "") for b in r.get("content", [])
                  if b.get("type") == "text")
    return _json_or_die(txt, r)


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
    required = schema.get("required") or []
    last = None
    for attempt in range(MAX_RETRY):
        try:
            result = parse(_post(url, body, headers))
            missing = [k for k in required if k not in result]
            if missing and len(missing) == len(required):
                # 하나도 안 맞으면 게이트웨이가 스키마를 통째로 무시한 것.
                # 재시도해도 같다.
                raise ParseError(
                    "스키마 미적용 — 필수 키 전부 누락. 받은 키 %r. "
                    "LLM_STRUCTURED 설정을 확인하라 (probe.py)."
                    % (sorted(result)[:8],))
            if missing:
                # 일부만 누락 = 모델 변덕. 한 번은 다시 물어본다.
                if attempt < MAX_RETRY - 1:
                    last = ParseError("키 누락 %r" % missing)
                    time.sleep(1)
                    continue
                raise ParseError(
                    "키 누락이 재시도 후에도 지속: %r (받은 키 %r)"
                    % (missing, sorted(result)[:10]))
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
        except ParseError:
            raise                                   # 재시도해도 동일
        except Exception as e:
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
