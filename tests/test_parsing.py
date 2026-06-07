"""Тесты разбора ответа модели: валидный JSON, обёрнутый в ```, мусор, невалидный тип."""
from types import SimpleNamespace

import pytest

from llm_client import LLMResponseError, _json_from_text, parse_response
from models import Extraction


def _resp(content: str):
    """Собрать объект-заглушку в форме ответа chat.completions."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_parse_plain_json():
    r = _resp('{"type":"order","product":"X","contacts":{"phone":"+7","email":null}}')
    e = parse_response(r)
    assert isinstance(e, Extraction)
    assert e.type == "order"
    assert e.contacts.phone == "+7"
    assert e.contacts.email is None


def test_parse_fenced_json():
    r = _resp('```json\n{"type":"question","product":null,"contacts":{"phone":null,"email":null}}\n```')
    e = parse_response(r)
    assert e.type == "question"
    assert e.product is None


def test_parse_with_surrounding_text():
    r = _resp('Вот результат: {"type":"complaint","product":"Y","contacts":{"phone":null,"email":"a@b.c"}} спасибо')
    e = parse_response(r)
    assert e.type == "complaint"
    assert e.contacts.email == "a@b.c"


def test_parse_invalid_type_raises():
    r = _resp('{"type":"spam","product":null,"contacts":{"phone":null,"email":null}}')
    with pytest.raises(LLMResponseError):
        parse_response(r)


def test_parse_not_json_raises():
    with pytest.raises(LLMResponseError):
        parse_response(_resp("no json here"))


def test_parse_empty_raises():
    with pytest.raises(LLMResponseError):
        parse_response(_resp(""))


def test_json_from_text_simple():
    assert _json_from_text('{"a": 1}') == {"a": 1}
