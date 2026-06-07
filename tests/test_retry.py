"""Тесты политики повторов: что считается транзиентным, а что — нет."""
import httpx
import openai

from llm_client import LLMResponseError, _is_retryable


def _status_error(code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://example.test")
    response = httpx.Response(code, request=request)
    return openai.APIStatusError("err", response=response, body=None)


def test_llm_response_error_is_retryable():
    assert _is_retryable(LLMResponseError("bad json")) is True


def test_unrelated_exception_not_retryable():
    assert _is_retryable(ValueError("oops")) is False


def test_server_5xx_is_retryable():
    assert _is_retryable(_status_error(503)) is True


def test_client_4xx_not_retryable():
    assert _is_retryable(_status_error(400)) is False


def test_connection_error_is_retryable():
    request = httpx.Request("POST", "http://example.test")
    assert _is_retryable(openai.APIConnectionError(request=request)) is True
