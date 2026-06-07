from __future__ import annotations

import asyncio
from typing import Callable, Optional

from llm_client import AsyncClassifierClient, ClassifierClient

ERROR_MESSAGE = "API request failed"
DEFAULT_CONCURRENCY = 5
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
        return _error_row(base)
    return _success_row(base, extraction)


async def classify_batch_async(
    client: AsyncClassifierClient,
    items: list[dict],
    concurrency: int = DEFAULT_CONCURRENCY,
    on_progress: Optional[ProgressCallback] = None,
) -> list[dict]:
    """Async version: runs up to ``concurrency`` requests at once.

    Order of results matches the order of ``items``. A bounded semaphore caps how
    many requests hit the API simultaneously so we don't trip rate limits.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    total = len(items)
    done = 0

    async def worker(item: dict) -> dict:
        nonlocal done
        async with semaphore:
            row = await _classify_item_async(client, item)
        done += 1  # single-threaded event loop → plain increment is safe
        if on_progress is not None:
            on_progress(done, total)
        return row

    return await asyncio.gather(*(worker(item) for item in items))


async def _classify_item_async(client: AsyncClassifierClient, item: dict) -> dict:
    base = {"id": item.get("id"), "text": item.get("text", "")}
    try:
        extraction = await client.classify(base["text"])
    except Exception:  # noqa: BLE001 - one failed message must not stop the batch
        return _error_row(base)
    return _success_row(base, extraction)


def _success_row(base: dict, extraction) -> dict:
    return {
        **base,
        "type": extraction.type,
        "product": extraction.product or "",
        "phone": extraction.contacts.phone or "",
        "email": extraction.contacts.email or "",
        "error": "",
    }


def _error_row(base: dict) -> dict:
    return {**base, "type": "", "product": "", "phone": "", "email": "", "error": ERROR_MESSAGE}


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
