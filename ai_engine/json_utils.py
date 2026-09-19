"""
ai_engine/json_utils.py
=========================
LLM 응답 텍스트에서 JSON만 잘라내는 파싱 방어 유틸리티.

matching_engine.py에서 검증된 로직을 동작 변경 없이 그대로 옮겼다. extract_profile,
generate_icebreakers, extract_delta 등 "LLM에게 JSON만 출력하라고 시키는" 모든 함수가
공통으로 이 모듈을 사용한다.
"""

from __future__ import annotations


def _extract_json_object(text: str) -> str:
    """
    무엇을: 모델 응답 앞뒤에 붙은 군더더기 텍스트(인사말, 설명, ```json 코드블록 등)를 잘라내고
    첫 '{'부터 마지막 '}'까지만 남겨 JSON 파싱 성공률을 높인다.

    왜 필요한가: 학교/기관 게이트웨이가 물려주는 모델은 "JSON만 출력해"라는 지시를 상용
    모델만큼 엄격히 지키지 않는 경우가 있다(응답 앞에 "네, 분석해드릴게요" 같은 문장을
    붙이거나 토큰 제한으로 응답이 중간에 잘리는 경우). json.loads()는 문자열이 정확히
    JSON이 아니면 바로 실패하므로, 파싱 전에 JSON처럼 보이는 부분만 잘라내는 방어 로직을
    둔다.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"JSON 객체를 찾을 수 없음: {text[:200]!r}")
    return text[start : end + 1]


def _extract_json_array(text: str) -> str:
    """모델이 JSON 배열 앞뒤로 군더더기 텍스트를 붙여도, 첫 '['부터 마지막 ']'까지만 잘라내 파싱 성공률을 높인다."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"JSON 배열을 찾을 수 없음: {text[:200]!r}")
    return text[start : end + 1]
