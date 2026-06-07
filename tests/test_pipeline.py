"""Тесты пакетной обработки (sync и async) на фейковых клиентах — без сети."""
from models import Contacts, Extraction
from pipeline import ERROR_MESSAGE, classify_batch, classify_batch_async, to_task_json


def _ext(type_: str, product, phone=None, email=None) -> Extraction:
    return Extraction(type=type_, product=product, contacts=Contacts(phone=phone, email=email))


class FakeSyncClient:
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def classify(self, text: str) -> Extraction:
        value = self.mapping[text]
        if isinstance(value, Exception):
            raise value
        return value


class FakeAsyncClient:
    def __init__(self, mapping: dict):
        self.mapping = mapping

    async def classify(self, text: str) -> Extraction:
        value = self.mapping[text]
        if isinstance(value, Exception):
            raise value
        return value


def test_classify_batch_success_and_error_isolation():
    mapping = {
        "ok": _ext("order", "ноутбук", phone="+7"),
        "boom": RuntimeError("network down"),
    }
    items = [{"id": 1, "text": "ok"}, {"id": 2, "text": "boom"}]
    progress: list[tuple[int, int]] = []

    results = classify_batch(FakeSyncClient(mapping), items, on_progress=lambda n, t: progress.append((n, t)))

    assert len(results) == 2
    assert results[0]["type"] == "order"
    assert results[0]["product"] == "ноутбук"
    assert results[0]["phone"] == "+7"
    assert results[0]["error"] == ""
    # одно упавшее сообщение не валит остальные — у него заполнен error
    assert results[1]["error"] == ERROR_MESSAGE
    assert results[1]["type"] == ""
    assert progress[-1] == (2, 2)  # прогресс дошёл до конца


def test_to_task_json_shapes():
    rows = [
        {"id": 1, "text": "t", "type": "order", "product": "X", "phone": "+7", "email": "", "error": ""},
        {"id": 2, "text": "t", "type": "", "product": "", "phone": "", "email": "", "error": ERROR_MESSAGE},
    ]
    out = to_task_json(rows)
    assert out[0] == {"id": 1, "type": "order", "product": "X", "contacts": {"phone": "+7", "email": None}}
    assert out[1] == {"id": 2, "error": ERROR_MESSAGE}


async def test_classify_batch_async_preserves_order():
    mapping = {f"m{i}": _ext("question", f"p{i}") for i in range(10)}
    items = [{"id": i, "text": f"m{i}"} for i in range(10)]

    results = await classify_batch_async(FakeAsyncClient(mapping), items, concurrency=4)

    # порядок результатов совпадает с порядком входа, несмотря на параллелизм
    assert [r["id"] for r in results] == list(range(10))
    assert [r["product"] for r in results] == [f"p{i}" for i in range(10)]
