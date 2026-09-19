"""
ai_engine/llm_client.py
=========================
LLM 백엔드 호출 추상화 계층.

matching_engine.py에서 이미 실제 게이트웨이(학교/기관 OpenAI 호환 프록시 및 순정 Anthropic
API)로 검증을 마친 _get_anthropic_client/_get_chat_backend/_call_llm 로직을 동작 변경 없이
그대로 옮긴 파일이다. ai_engine의 다른 모든 모듈(profile.py, matching.py, feedback.py)은
LLM 호출이 필요할 때 이 모듈의 _call_llm()만 사용한다 - Anthropic 순정 API인지, 학교/기관의
OpenAI 호환 게이트웨이인지는 이 모듈 안에서만 알면 되고 나머지 코드는 신경 쓸 필요가 없다.

환경변수:
    - ANTHROPIC_API_KEY  : 순정 Anthropic API 키. ANTHROPIC_BASE_URL을 같이 주면
                           /v1/messages 스키마를 쓰는 호환 게이트웨이도 지원한다.
    - LLM_BASE_URL       : 대학/기관 프록시처럼 OpenAI Chat Completions 스키마
                           (/v1/chat/completions)로 Claude를 감싸 제공하는 게이트웨이의
                           Base URL. 설정돼 있으면 이 방식이 최우선으로 사용된다.
    - LLM_API_KEY        : LLM_BASE_URL용 키(없으면 ANTHROPIC_API_KEY를 재사용).
    - ANTHROPIC_MODEL    : 두 모드 공통으로 사용할 모델 이름(기본값: claude-sonnet-5).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

# --- 선택적 외부 패키지 -------------------------------------------------
# 패키지가 설치되어 있지 않거나 API 키가 없어도 데모/테스트가 죽지 않도록
# import 실패를 흡수하고, 실제 사용 시점에 None 체크로 폴백 경로를 탄다.
try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # type: ignore

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai_engine.llm_client")

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


def _get_anthropic_client() -> Optional["anthropic.Anthropic"]:
    """
    환경변수에 키가 있고 패키지가 설치돼 있을 때만 클라이언트를 생성, 그 외엔 None을 반환한다.

    ANTHROPIC_BASE_URL이 설정되어 있으면 그 주소로 요청을 보낸다. 학교/기관 플랫폼이
    발급한 키는 api.anthropic.com이 아니라 자체 게이트웨이(프록시) 엔드포인트를 쓰는 경우가
    많으므로, 그런 경우 플랫폼이 안내하는 Base URL을 이 환경변수에 지정하면 된다.
    """
    if anthropic is None:
        logger.warning("anthropic 패키지가 설치되어 있지 않습니다. 규칙 기반 폴백을 사용합니다.")
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY 환경변수가 설정되어 있지 않습니다. 규칙 기반 폴백을 사용합니다.")
        return None
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    try:
        if base_url:
            return anthropic.Anthropic(api_key=api_key, base_url=base_url)
        return anthropic.Anthropic(api_key=api_key)
    except Exception as exc:  # pragma: no cover - 방어적 처리
        logger.error(f"Anthropic 클라이언트 생성 실패: {exc}")
        return None


def _get_chat_backend() -> Optional[Tuple[str, Any]]:
    """
    무엇을: 실제 채팅형 LLM 호출에 사용할 백엔드를 결정해 (모드, 클라이언트) 튜플로 반환한다.

    왜 두 가지 모드가 필요한가:
    - 순정 Anthropic API(api.anthropic.com, 스키마: POST /v1/messages,
      응답은 content[0].text)를 직접 쓰는 경우 anthropic 패키지를 사용한다.
    - 대학/기관에서 흔히 제공하는 LiteLLM류 프록시/게이트웨이는 비용 관리를 위해
      Claude 모델을 OpenAI Chat Completions 스키마(POST /v1/chat/completions,
      응답은 choices[0].message.content)로 감싸서 제공하는 경우가 많다. 이 경우
      Anthropic SDK로는 호출할 수 없으므로, LLM_BASE_URL 환경변수가 설정되어 있으면
      openai 패키지로 같은 모델을 OpenAI 호환 방식으로 호출한다.

    우선순위: LLM_BASE_URL(OpenAI 호환 게이트웨이) > ANTHROPIC_API_KEY(순정 Anthropic SDK).
    """
    llm_base_url = os.environ.get("LLM_BASE_URL")
    if llm_base_url:
        if openai is None:
            logger.warning("LLM_BASE_URL이 설정되어 있지만 openai 패키지가 설치되어 있지 않습니다.")
            return None
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("LLM_BASE_URL은 설정되었지만 LLM_API_KEY/ANTHROPIC_API_KEY가 없습니다.")
            return None
        try:
            client = openai.OpenAI(api_key=api_key, base_url=llm_base_url)
            return ("openai_compatible", client)
        except Exception as exc:
            logger.error(f"OpenAI 호환 게이트웨이 클라이언트 생성 실패: {exc}")
            return None

    anthropic_client = _get_anthropic_client()
    if anthropic_client is not None:
        return ("anthropic", anthropic_client)

    return None


def _call_llm(user_prompt: str, max_tokens: int, system_prompt: Optional[str] = None) -> str:
    """
    무엇을: _get_chat_backend()가 고른 백엔드에 맞춰 실제 LLM 호출을 수행하고 응답 텍스트만
    뽑아 반환한다. 두 스키마(Anthropic Messages / OpenAI Chat Completions) 차이를 이 함수
    안에 가둬서, 호출하는 쪽(extract_profile/generate_match_reason/generate_icebreakers/
    extract_delta 등)은 백엔드가 무엇이든 신경 쓰지 않고 "프롬프트 -> 텍스트"만 다루도록 한다.

    실패 시 예외를 그대로 전파한다 - 호출부가 이미 try/except로 감싸서 규칙 기반 폴백을
    준비해 두었기 때문에, 여기서 흡수하지 않고 올려보내는 편이 원인 파악에 유리하다.
    """
    backend = _get_chat_backend()
    if backend is None:
        raise RuntimeError("사용 가능한 LLM 백엔드가 없습니다(API 키/게이트웨이 설정을 확인하세요).")
    mode, client = backend

    if mode == "anthropic":
        kwargs: Dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()

    # mode == "openai_compatible": 학교/기관 프록시 등 OpenAI Chat Completions 스키마.
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content.strip()
