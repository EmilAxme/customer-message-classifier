"""Веб-приложение для классификации обращений клиентов.

Запуск: streamlit run app.py  (или двойной клик по «Запустить.command»).
Человеку без знания кода: вставьте/загрузите обращения → «Обработать» →
скачайте таблицу с результатами.
"""
from __future__ import annotations

import asyncio
import io
import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm_client import AsyncClassifierClient
from pipeline import DEFAULT_CONCURRENCY, classify_batch_async, to_task_json

load_dotenv()

TYPE_RU = {"order": "Заказ", "complaint": "Жалоба", "question": "Вопрос"}
COLUMNS_RU = {
    "id": "№",
    "text": "Текст обращения",
    "type": "Тип",
    "product": "Товар",
    "phone": "Телефон",
    "email": "E-mail",
    "error": "Статус",
}
TEXT_HINTS = ("text", "message", "обращение", "текст", "сообщение")

st.set_page_config(page_title="Классификатор обращений", page_icon="📨", layout="wide")


# ---------- чтение загруженного файла ----------
def items_from_file(uploaded) -> list[dict]:
    """Вернуть список {id, text} из загруженного файла (txt/json/csv/excel)."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()

    if name.endswith(".txt"):
        lines = data.decode("utf-8", errors="replace").splitlines()
        return [{"id": i, "text": s.strip()} for i, s in enumerate(lines, 1) if s.strip()]

    if name.endswith(".json"):
        parsed = json.loads(data.decode("utf-8", errors="replace"))
        items = []
        for i, el in enumerate(parsed, 1):
            if isinstance(el, dict):
                items.append({"id": el.get("id", i), "text": str(el.get("text", "")).strip()})
            else:
                items.append({"id": i, "text": str(el).strip()})
        return [it for it in items if it["text"]]

    # табличные форматы
    df = pd.read_excel(io.BytesIO(data)) if name.endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(data))
    if df.empty:
        return []
    columns = list(df.columns)
    guess = next((c for c in columns if str(c).strip().lower() in TEXT_HINTS), columns[0])
    text_col = st.selectbox("Колонка с текстом обращения", columns, index=columns.index(guess))
    id_options = ["(номер строки)"] + columns
    id_choice = st.selectbox("Колонка с ID (необязательно)", id_options, index=0)

    items = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        text = str(row[text_col]).strip()
        if not text or text.lower() == "nan":
            continue
        msg_id = i if id_choice == "(номер строки)" else row[id_choice]
        items.append({"id": msg_id, "text": text})
    return items


# ---------- оформление таблицы и выгрузки ----------
def to_display_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "type": "—" if r["error"] else TYPE_RU.get(r["type"], r["type"]),
                "product": r["product"],
                "phone": r["phone"],
                "email": r["email"],
                "error": "Ошибка обработки" if r["error"] else "OK",
            }
        )
    return pd.DataFrame(rows).rename(columns=COLUMNS_RU)


def excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Результат")
    return buffer.getvalue()


# ---------- интерфейс ----------
st.title("📨 Классификатор обращений клиентов")
st.caption(
    "Определяет тип обращения (заказ / жалоба / вопрос) и достаёт из текста "
    "товар, телефон и e-mail. Подходит для разбора любого количества сообщений."
)

with st.sidebar:
    st.header("⚙️ Настройки доступа")
    api_key = st.text_input("API-ключ", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    base_url = st.text_input("Адрес сервера", value=os.getenv("OPENAI_BASE_URL", "https://codex.sale/v1"))
    model = st.text_input("Модель", value=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    concurrency = st.slider(
        "Одновременных запросов", min_value=1, max_value=20, value=DEFAULT_CONCURRENCY,
        help="Сколько обращений обрабатывать параллельно. Больше — быстрее, но выше нагрузка на сервис.",
    )
    st.caption("Обычно ключ уже прописан в файле .env. Менять эти поля не нужно.")

st.subheader("Шаг 1. Загрузите обращения")
items: list[dict] = []
tab_text, tab_file = st.tabs(["✍️ Вставить текст", "📄 Загрузить файл"])

with tab_text:
    raw = st.text_area(
        "Каждое обращение — с новой строки",
        height=200,
        placeholder=(
            "Купил наушники Sony WH-1000XM5, не работают. Почта ivan@example.com\n"
            "Хочу заказать кофемашину DeLonghi. Тел +79991234567\n"
            "Подскажите, есть ли в наличии iPhone 15 Pro?"
        ),
    )
    if raw.strip():
        items = [{"id": i, "text": s.strip()} for i, s in enumerate(raw.splitlines(), 1) if s.strip()]

with tab_file:
    uploaded = st.file_uploader("CSV, Excel, TXT или JSON", type=["csv", "xlsx", "xls", "txt", "json"])
    if uploaded is not None:
        try:
            file_items = items_from_file(uploaded)
            if file_items:
                items = file_items
                st.success(f"Загружено обращений: {len(items)}")
            else:
                st.warning("В файле не найдено текста обращений.")
        except Exception as exc:  # noqa: BLE001 - показать понятную ошибку пользователю
            st.error(f"Не удалось прочитать файл: {exc}")

st.subheader("Шаг 2. Обработка")
if st.button("🚀 Обработать", type="primary", disabled=not items):
    if not api_key:
        st.error("Введите API-ключ в настройках слева.")
    else:
        bar = st.progress(0.0, text="Обработка…")

        async def run() -> list[dict]:
            client = AsyncClassifierClient(api_key=api_key, model=model, base_url=base_url)
            try:
                return await classify_batch_async(
                    client, items, concurrency=concurrency,
                    on_progress=lambda n, total: bar.progress(n / total, text=f"Обработано {n} из {total}"),
                )
            finally:
                await client.aclose()

        results = asyncio.run(run())
        bar.empty()
        failed = sum(1 for r in results if r["error"])
        st.session_state["results"] = results
        if failed:
            st.warning(f"Готово. Успешно: {len(results) - failed}, с ошибкой: {failed}.")
        else:
            st.success(f"Готово! Обработано обращений: {len(results)}.")

if "results" in st.session_state:
    results = st.session_state["results"]
    df = to_display_df(results)

    st.subheader("Шаг 3. Результат")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Шаг 4. Скачать")
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "⬇️ Excel (.xlsx)", excel_bytes(df), "результат.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    col2.download_button(
        "⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"), "результат.csv", "text/csv"
    )
    col3.download_button(
        "⬇️ JSON",
        json.dumps(to_task_json(results), ensure_ascii=False, indent=2).encode("utf-8"),
        "результат.json",
        "application/json",
    )
