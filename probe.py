"""게이트웨이가 각 방식에 어떻게 응답하는지 확인한다.

    source .env && python3 probe.py

세 가지를 순서대로 시도하고, 성공한 방식을 알려준다.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ["LLM_BASE_URL"].rstrip("/")
KEY = os.environ["LLM_API_KEY"]
MODEL = os.environ["LLM_MODEL"]

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"extraversion": {"type": "integer",
                                    "minimum": 1, "maximum": 5},
                   "interests": {"type": "array",
                                 "items": {"type": "string"},
                                 "minItems": 3, "maxItems": 3}},
    "required": ["extraversion", "interests"],
}
Q = ("다음 사람을 외향성 1~5점으로 채점하고 관심사 3개를 뽑아 JSON으로만 답하라.\n"
     "'혼자 있을 때 에너지가 차요. 요즘 홈서버랑 리눅스에 빠졌고 주말엔 등산해요.'")


def post(url, body, headers):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def show(name, fn):
    print("\n" + "=" * 60)
    print("[%s]" % name)
    try:
        r = fn()
        print(json.dumps(r, ensure_ascii=False)[:1000])
    except urllib.error.HTTPError as e:
        print("HTTP %s: %s" % (e.code, e.read().decode()[:500]))
    except Exception as e:
        print("실패: %r" % e)


msgs = [{"role": "user", "content": Q}]

show("A. openai + json_schema", lambda: post(
    BASE + "/chat/completions",
    {"model": MODEL, "messages": msgs, "temperature": 0,
     "response_format": {"type": "json_schema",
                         "json_schema": {"name": "out", "strict": True,
                                         "schema": SCHEMA}}},
    {"Authorization": "Bearer " + KEY}))

show("B. openai + json_object", lambda: post(
    BASE + "/chat/completions",
    {"model": MODEL, "messages": msgs, "temperature": 0,
     "response_format": {"type": "json_object"}},
    {"Authorization": "Bearer " + KEY}))

show("C. openai + tools", lambda: post(
    BASE + "/chat/completions",
    {"model": MODEL, "messages": msgs, "temperature": 0,
     "tools": [{"type": "function",
                "function": {"name": "emit", "parameters": SCHEMA}}],
     "tool_choice": {"type": "function", "function": {"name": "emit"}}},
    {"Authorization": "Bearer " + KEY}))

show("D. anthropic /messages + tool_use", lambda: post(
    BASE + "/messages",
    {"model": MODEL, "max_tokens": 1024, "temperature": 0,
     "messages": msgs,
     "tools": [{"name": "emit", "input_schema": SCHEMA}],
     "tool_choice": {"type": "tool", "name": "emit"}},
    {"x-api-key": KEY, "anthropic-version": "2023-06-01",
     "Authorization": "Bearer " + KEY}))

show("E. 평문 (형식 지정 없음)", lambda: post(
    BASE + "/chat/completions",
    {"model": MODEL, "messages": msgs, "temperature": 0},
    {"Authorization": "Bearer " + KEY}))
