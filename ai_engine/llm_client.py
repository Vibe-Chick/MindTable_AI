"""
ai_engine/llm_client.py
=========================
LLM 호출 계층 — 팀 통합 결정(BRANCH_COMPARISON.md)에 따라 김주환 브랜치의
`AI/src/llm.py`를 이식한 버전. 캐시·재시도·게이트웨이 스키마 무시 감지 등
실전에서 검증된 로직을 그대로 가져왔다.

jaebin 대비 변경점:
    - 디스크 캐시 추가: 프롬프트+스키마+모델을 해시한 키로 결과를 cache/*.json에
      저장한다. 재실행이 공짜가 되고, 무대에서 LLM_CACHE_ONLY=1이면 캐시 미스를
      에러로 즉시 중단시켜 "실수로 실제 API를 호출하는 상황" 자체를 차단한다.
    - 재시도가 HTTP 상태코드 기반이다: 429/5xx만 재시도, 4xx는 재시도해도 같은
      결과이므로 즉시 실패. "필수 키가 전부 누락"이면 게이트웨이가 구조화 출력
      스키마 자체를 무시한 것이므로 재시도하지 않고 바로 예외를 던진다(재시도해도
      같은 결과가 나오기 때문). "일부만 누락"이면 모델 변덕으로 보고 한 번 더
      물어본다.
    - 구조화 출력 방식을 tools(기본)/json_schema/json_object 중 선택 가능하게
      했다. 국민대 게이트웨이는 json_schema/json_object를 에러 없이 무시하고
      tools 방식만 실제로 동작한다는 사실이 probe.py로 검증되었다
      (AI/README.md, AI/probe.py 참고).
    - openai/anthropic/gemini 3개 프로바이더를 지원한다.

jaebin에서 유지한 것 (팀 결정: "게이트웨이 경로가 실패해도 완전히 죽지 않게"):
    - 순정 Anthropic API 직접 호출 fallback: 기본 프로바이더 경로가 모두
      실패하면(파싱 실패 제외 — 파싱 실패는 재시도해도 같은 결과이므로
      폴백하지 않는다), ANTHROPIC_API_KEY가 있을 경우 api.anthropic.com으로
      한 번 더 시도한다.
    - 환경변수 별칭: LLM_API_KEY/LLM_BASE_URL/LLM_MODEL이 우선이지만, 기존
      ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL/ANTHROPIC_MODEL도 그대로 인식한다.
      팀원 누구도 .env를 급하게 다시 세팅할 필요가 없다.

환경변수:
    LLM_PROVIDER   openai | anthropic | gemini
                   명시하지 않으면: LLM_BASE_URL이 있으면 openai(게이트웨이가
                   보통 OpenAI 호환), 없고 ANTHROPIC_API_KEY만 있으면 anthropic,
                   그 외 기본값 openai.
    LLM_MODEL      모델 ID (별칭: ANTHROPIC_MODEL)
    LLM_API_KEY    API 키 (별칭: ANTHROPIC_API_KEY)
    LLM_BASE_URL   게이트웨이 주소 (별칭: ANTHROPIC_BASE_URL)
    LLM_CACHE_ONLY=1   캐시에 없으면 에러. 무대 시연 때 반드시 켠다.
    LLM_STRUCTURED     tools(기본) | json_schema | json_object
    LLM_DEBUG=1        원시 응답을 stderr에 덤프.
    LLM_STUB=1         API 없이 스키마에 맞는 가짜 응답 생성(개발 전용,
                       캐시에 쓰지 않는다).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

logger = logging.getLogger("ai_engine.llm_client")

# ai_engine/ 의 부모(mind_table/)에 cache/ 를 둔다.
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)


def _env(*names: str, default: str = "") -> str:
    """앞에서부터 먼저 설정된 환경변수 값을 쓴다 (별칭 처리)."""
    for name in names:
        v = os.environ.get(name)
        if v:
            return v
    return default


CACHE_ONLY = os.environ.get("LLM_CACHE_ONLY", "") == "1"
STUB = os.environ.get("LLM_STUB", "") == "1"
DEBUG = os.environ.get("LLM_DEBUG", "") == "1"
STRUCTURED = os.environ.get("LLM_STRUCTURED", "tools").lower()
_stub_warned = False

DEFAULT_MODEL = {
    "openai": _env("LLM_MODEL", "ANTHROPIC_MODEL", default="claude-sonnet-5"),
    "anthropic": _env("LLM_MODEL", "ANTHROPIC_MODEL", default="claude-sonnet-5"),
    "gemini": "SET_LLM_MODEL",
}

DEFAULT_BASE = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}


def _infer_provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER")
    if explicit:
        return explicit.lower()
    if os.environ.get("LLM_BASE_URL"):
        return "openai"  # 학교/기관 게이트웨이는 보통 OpenAI 호환 스키마
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("LLM_API_KEY"):
        return "anthropic"  # 기존 jaebin 스타일: 순정 Anthropic 키만 있는 경우
    return "openai"


PROVIDER = _infer_provider()
API_KEY = _env("LLM_API_KEY", "ANTHROPIC_API_KEY")
MODEL = _env("LLM_MODEL", "ANTHROPIC_MODEL") or DEFAULT_MODEL.get(PROVIDER, "")
BASE_URL = (_env("LLM_BASE_URL", "ANTHROPIC_BASE_URL")
            or DEFAULT_BASE.get(PROVIDER, "")).rstrip("/")

# 순정 Anthropic 폴백용 (기본 프로바이더가 anthropic이 아닐 때만 사용).
_FALLBACK_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_FALLBACK_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

MAX_RETRY = 3
WORKERS = int(os.environ.get("LLM_WORKERS", "4"))  # 신규 키는 동시 요청 많으면 429


class LLMError(RuntimeError):
    pass


class ParseError(LLMError):
    """응답은 왔는데 형식이 다르다. 재시도해도 같은 결과이므로 즉시 중단."""


# --- 캐시 -----------------------------------------------------------------

def _cache_key(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + ".json")


def cache_stats() -> Dict[str, Any]:
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    return {"entries": len(files), "dir": CACHE_DIR}


# --- 벤더별 요청/응답 --------------------------------------------------------

def _post(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def _strip_fence(t: Optional[str]) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _json_or_die(txt: Optional[str], raw: Dict[str, Any]) -> Dict[str, Any]:
    txt = _strip_fence(txt)
    if not txt:
        raise ParseError("빈 응답. 원시: " + json.dumps(raw, ensure_ascii=False)[:600])
    try:
        return json.loads(txt)
    except Exception:
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(txt[i:j + 1])
            except Exception:
                pass
        raise ParseError("JSON 아님: " + txt[:400])


def _parse_openai(r: Dict[str, Any]) -> Dict[str, Any]:
    if DEBUG:
        sys.stderr.write("[llm] raw: %s\n" % json.dumps(r, ensure_ascii=False)[:1200])
    if "choices" not in r:
        raise ParseError("choices 없음. 원시: " + json.dumps(r, ensure_ascii=False)[:600])
    msg = r["choices"][0].get("message", {}) or {}
    txt = msg.get("content")
    if isinstance(txt, list):
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


def _parse_anthropic(r: Dict[str, Any]) -> Dict[str, Any]:
    if DEBUG:
        sys.stderr.write("[llm] raw: %s\n" % json.dumps(r, ensure_ascii=False)[:1200])
    for b in r.get("content", []):
        if b.get("type") == "tool_use":
            return b["input"]
    txt = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
    return _json_or_die(txt, r)


def _build(provider: str, model: str, base_url: str, api_key: str,
           prompt: str, schema: Dict[str, Any], system: Optional[str], temp: float):
    """벤더별 요청 본문 + 응답 파서를 함께 돌려준다."""
    if provider == "openai":
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body: Dict[str, Any] = {"model": model, "messages": msgs, "temperature": temp}

        if STRUCTURED == "tools":
            body["tools"] = [{"type": "function",
                              "function": {"name": "emit", "description": "결과 출력",
                                           "parameters": schema}}]
            body["tool_choice"] = {"type": "function", "function": {"name": "emit"}}
        elif STRUCTURED == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True, "schema": schema}}
        elif STRUCTURED == "json_object":
            body["response_format"] = {"type": "json_object"}

        return (base_url + "/chat/completions", body,
                {"Authorization": "Bearer " + api_key}, _parse_openai)

    if provider == "anthropic":
        body = {
            "model": model, "max_tokens": 1024, "temperature": temp,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"name": "emit", "description": "결과 출력", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "emit"},
        }
        if system:
            body["system"] = system
        return (base_url + "/messages", body,
                {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                _parse_anthropic)

    if provider == "gemini":
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
        url = base_url + "/models/" + model + ":generateContent?key=" + api_key
        return (url, body, {},
                lambda r: json.loads(r["candidates"][0]["content"]["parts"][0]["text"]))

    raise LLMError("unknown LLM_PROVIDER: " + provider)


# --- STUB (개발용 가짜 응답) -------------------------------------------------

_STUB_WORDS = ["등산", "홈서버", "재즈", "베이킹", "독립영화", "보드게임",
               "배낭여행", "클라이밍", "사진", "철학"]


def _stub_value(spec: Dict[str, Any], seed: int):
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
                for i, (k, v) in enumerate(sorted(spec.get("properties", {}).items()))}
    if t == "array":
        n = spec.get("minItems", 3)
        item = spec.get("items", {})
        if item.get("type") == "string":
            return rnd.sample(_STUB_WORDS, min(n, len(_STUB_WORDS)))
        return [_stub_value(item, seed + i) for i in range(n)]
    return "(stub)"


def _stub(schema: Dict[str, Any], seed: int) -> Dict[str, Any]:
    global _stub_warned
    if not _stub_warned:
        sys.stderr.write("\n*** LLM_STUB=1 — 가짜 응답입니다. 결과 수치를 신뢰하지 마세요. ***\n\n")
        _stub_warned = True
    props = schema.get("properties", {})
    return {k: _stub_value(v, seed + i) for i, (k, v) in enumerate(sorted(props.items()))}


# --- 핵심 호출 ---------------------------------------------------------------

def _call_with(provider: str, model: str, base_url: str, api_key: str,
               prompt: str, schema: Dict[str, Any], system: Optional[str],
               temp: float, tag: str) -> Dict[str, Any]:
    """지정된 (provider, model, base_url, api_key) 조합으로 캐시+재시도까지 포함해 1회 호출."""
    key = _cache_key({"p": prompt, "s": schema, "sys": system, "t": temp,
                      "m": model, "v": provider})
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["result"]

    if CACHE_ONLY:
        raise LLMError(
            "LLM_CACHE_ONLY=1 인데 캐시에 없음 (tag=%s, key=%s). "
            "무대 시연 전에 전체 파이프라인을 한 번 돌려 캐시를 채울 것." % (tag, key))
    if not api_key:
        raise LLMError("API 키 미설정 (provider=%s)" % provider)
    if model.startswith("SET_"):
        raise LLMError("모델 미설정 (provider=%s)" % provider)

    url, body, headers, parse = _build(provider, model, base_url, api_key,
                                       prompt, schema, system, temp)
    required = schema.get("required") or []
    last = None
    for attempt in range(MAX_RETRY):
        try:
            result = parse(_post(url, body, headers))
            missing = [k for k in required if k not in result]
            if missing and len(missing) == len(required):
                raise ParseError(
                    "스키마 미적용 — 필수 키 전부 누락. 받은 키 %r. "
                    "LLM_STRUCTURED 설정을 확인하라." % (sorted(result)[:8],))
            if missing:
                if attempt < MAX_RETRY - 1:
                    last = ParseError("키 누락 %r" % missing)
                    time.sleep(1)
                    continue
                raise ParseError("키 누락이 재시도 후에도 지속: %r (받은 키 %r)"
                                 % (missing, sorted(result)[:10]))
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"tag": tag, "model": model, "prompt": prompt, "result": result},
                         f, ensure_ascii=False, indent=2)
            return result
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529):
                time.sleep((2 ** attempt) + random.random())
                continue
            raise LLMError("HTTP %s: %s" % (e.code, e.read()[:400])) from e
        except ParseError:
            raise  # 재시도해도 동일
        except Exception as e:
            last = e
            time.sleep(1 + attempt)
    raise LLMError("재시도 %d회 실패 (tag=%s): %r" % (MAX_RETRY, tag, last))


def call(prompt: str, schema: Dict[str, Any], system: Optional[str] = None,
         temp: float = 0.0, tag: str = "") -> Dict[str, Any]:
    """구조화 출력 1회. 캐시 히트면 API를 건드리지 않는다.

    기본 프로바이더 경로가 (파싱 실패가 아닌 이유로) 완전히 실패하고,
    ANTHROPIC_API_KEY가 설정돼 있고, 기본 프로바이더가 anthropic이 아니면
    순정 Anthropic API로 한 번 더 시도한다 (게이트웨이 장애에도 완전히
    죽지 않도록 하는 팀 결정 사항).
    """
    if STUB:
        key = _cache_key({"p": prompt, "s": schema, "sys": system, "t": temp,
                          "m": MODEL, "v": PROVIDER})
        return _stub(schema, int(key[:8], 16))

    try:
        return _call_with(PROVIDER, MODEL, BASE_URL, API_KEY, prompt, schema, system, temp, tag)
    except ParseError:
        raise  # 재시도해도 동일한 결과이므로 폴백 의미 없음
    except LLMError as primary_exc:
        if PROVIDER != "anthropic" and _FALLBACK_KEY:
            logger.warning(
                "[llm_client] 기본 프로바이더(%s) 호출 실패, 순정 Anthropic API로 폴백 "
                "시도 (tag=%s): %s", PROVIDER, tag, primary_exc)
            try:
                return _call_with("anthropic", _FALLBACK_MODEL,
                                  DEFAULT_BASE["anthropic"], _FALLBACK_KEY,
                                  prompt, schema, system, temp, tag + ":fallback")
            except Exception as fallback_exc:
                logger.error("[llm_client] Anthropic 폴백도 실패 (tag=%s): %s",
                             tag, fallback_exc)
        raise


def map_call(items, fn, workers: int = WORKERS):
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
    """게이트웨이가 실제로 내주는 모델 목록."""
    req = urllib.request.Request(
        BASE_URL + "/models",
        headers={"Authorization": "Bearer " + API_KEY, "x-api-key": API_KEY})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    return sorted(m.get("id", "?") for m in data.get("data", []))


def is_configured() -> bool:
    """호출 가능한 백엔드(실키 또는 스텁)가 있는지."""
    return bool(STUB or API_KEY or _FALLBACK_KEY)
