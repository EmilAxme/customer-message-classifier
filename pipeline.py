from __future__ import annotations

from typing import Callable, Optional

from llm_client import ClassifierClient

ERROR_MESSAGE = "API request failed"
ProgressCallback = Callable[[int, int], None]


def classify_batch(
    client: ClassifierClient,
    items: list[dict],
    on_progress: Optional[ProgressCallback] = None,
) -> list[dict]:
    """Classify many messages, returning one flat result dict per item.

    ``items`` is a list of ``{"id": ..., "text": ...}``. Each result is a flat
    dict (id, text, type, product, phone, email, error) — convenient for tables
    and spreadsheets. A per-message failure is recorded in ``error`` instead of
    aborting the whole batch.
    """
    results: list[dict] = []
    total = len(items)
    for number, item in enumerate(items, start=1):
        results.append(_classify_item(client, item))
        if on_progress is not None:
            on_progress(number, total)
    return results


def _classify_item(client: ClassifierClient, item: dict) -> dict:
    base = {"id": item.get("id"), "text": item.get("text", "")}
    try:
        extraction = client.classify(base["text"])
    except Exception:  # noqa: BLE001 - one failed message must not stop the batch
        return {**base, "type": "", "product": "", "phone": "", "email": "", "error": ERROR_MESSAGE}
    return {
        **base,
        "type": extraction.type,
        "product": extraction.product or "",
        "phone": extraction.contacts.phone or "",
        "email": extraction.contacts.email or "",
        "error": "",
    }


def to_task_json(results: list[dict]) -> list[dict]:
    """Convert flat results to the original task JSON shape (nested contacts)."""
    out: list[dict] = []
    for row in results:
        if row.get("error"):
            out.append({"id": row["id"], "error": row["error"]})
        else:
            out.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "product": row["product"] or None,
                    "contacts": {
                        "phone": row["phone"] or None,
                        "email": row["email"] or None,
                    },
                }
            )
    return out
