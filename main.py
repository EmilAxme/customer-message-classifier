from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import ClassifierClient
from models import CustomerMessage

logger = logging.getLogger("classifier")

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "input.json"
OUTPUT_FILE = BASE_DIR / "output.json"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://codex.sale/v1"
ERROR_MESSAGE = "API request failed"
# Seconds to wait between requests, to stay under a provider's rate limit.
# 0 = no pacing (burst). Raise it if the endpoint returns 429s.
DEFAULT_REQUEST_INTERVAL = 0.0


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
    # Silence the SDK's per-request chatter; keep our own classifier logs.
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def load_messages(path: Path) -> list[CustomerMessage]:
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return [CustomerMessage.model_validate(item) for item in raw]


def process(client: ClassifierClient, message: CustomerMessage) -> dict:
    try:
        extraction = client.classify(message.text)
    except Exception:  # noqa: BLE001 - emit a uniform error record per message
        logger.exception("Failed to classify message id=%s", message.id)
        return {"id": message.id, "error": ERROR_MESSAGE}
    logger.info("Classified message id=%s as '%s'", message.id, extraction.type)
    return {"id": message.id, **extraction.model_dump()}


def main() -> int:
    configure_logging()
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    if not api_key:
        logger.error("OPENAI_API_KEY is not set (see .env.example)")
        return 1

    try:
        messages = load_messages(INPUT_FILE)
    except (OSError, json.JSONDecodeError, ValidationError):
        logger.exception("Could not load input file %s", INPUT_FILE)
        return 1

    interval = float(os.getenv("REQUEST_INTERVAL_SECONDS", DEFAULT_REQUEST_INTERVAL))
    logger.info("Loaded %d messages; using model '%s' @ %s", len(messages), model, base_url)
    client = ClassifierClient(api_key=api_key, model=model, base_url=base_url)

    results = []
    for index, message in enumerate(messages):
        if index and interval > 0:  # pace requests, but no delay before the first
            time.sleep(interval)
        results.append(process(client, message))

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    logger.info("Wrote %d results to %s", len(results), OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
