from __future__ import annotations

import json
import logging

import openai
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from models import Extraction
from prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 4
SERVER_ERROR_STATUS = 500


class LLMResponseError(Exception):
    """Raised when a response cannot be parsed into a valid Extraction."""


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient failures only; permanent ones (bad request, auth) fail fast."""
    if isinstance(exc, LLMResponseError):
        return True
    if isinstance(
        exc,
        (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, openai.APIStatusError):  # other HTTP errors: retry only 5xx
        return exc.status_code >= SERVER_ERROR_STATUS
    return False


# One retry policy, reused by both the sync and async clients.
_retry = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def parse_response(response) -> Extraction:
    """Turn a chat-completion response into a validated Extraction."""
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise LLMResponseError("response contained no content")
    payload = _json_from_text(content)
    try:
        return Extraction.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseError(f"response failed schema validation: {exc}") from exc


def _json_from_text(content: str) -> dict:
    """Parse a JSON object from the model output, tolerating ```-fenced text."""
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"response was not valid JSON: {exc}") from exc
    raise LLMResponseError("response did not contain a JSON object")


class ClassifierClient:
    """Synchronous client over an OpenAI-compatible chat endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        # Disable the SDK's built-in retries; tenacity owns the retry policy.
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._model = model

    def classify(self, text: str) -> Extraction:
        return self._classify(text)

    @_retry
    def _classify(self, text: str) -> Extraction:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=_build_messages(text),
        )
        return parse_response(response)


class AsyncClassifierClient:
    """Asynchronous client — lets many requests run concurrently."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._model = model

    async def classify(self, text: str) -> Extraction:
        return await self._classify(text)

    @_retry
    async def _classify(self, text: str) -> Extraction:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=_build_messages(text),
        )
        return parse_response(response)

    async def aclose(self) -> None:
        await self._client.close()
