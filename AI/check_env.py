"""환경 확인.  source .env && python3 check_env.py"""
import os
import sys

sys.path.insert(0, "src")
import llm  # noqa: E402

k = os.environ.get("LLM_API_KEY", "")
print("provider :", llm.PROVIDER)
print("base_url :", llm.BASE_URL)
print("model    :", llm.MODEL)
print("key      :", (k[:6] + "..." + k[-4:] + " (%d자)" % len(k)) if k else "없음")
print()
try:
    ms = llm.list_models()
    print("사용 가능한 모델 %d개:" % len(ms))
    for m in ms:
        print("  -", m)
    emb = [m for m in ms if "embed" in m.lower() or "bge" in m.lower()]
    print("\n임베딩 모델:", emb if emb else "없음 → 폴백 사용")
except Exception as e:
    print("모델 목록 조회 실패:", repr(e))
