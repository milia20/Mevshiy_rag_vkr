import gc
import gzip
import json
from pathlib import Path
from typing import cast

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from pandas import DataFrame


def df_to_list_of_dicts(df: pd.DataFrame) -> list[dict]:
    """
    Преобразует pandas DataFrame в список словарей.

    Каждый словарь соответствует одной строке DataFrame,
    где ключи — названия колонок, значения — данные в строке.

    Args:
        df: Входной DataFrame

    Returns:
        Список словарей
    """
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


empty_questions_df = DataFrame(
    columns=["dataset", "question_id", "question", "context", "correct_answer", "context_used"]
)


def load_ru_rag_questions(datasets_dir: Path, context_used: bool = False) -> DataFrame | None:
    pkl_path = datasets_dir / "ru_rag_test_dataset" / "ru_rag_test_dataset.pkl"
    if not pkl_path.exists():
        return empty_questions_df

    df = DataFrame(pd.read_pickle(pkl_path))
    df["dataset"] = "RuBQ"
    df = df.reset_index(names="question_id").rename(
        columns={"Вопрос": "question", "Правильный ответ": "correct_answer"}
    )
    df["context_used"] = context_used
    df["context"] = df["Контекст"] if context_used else None
    return df[["dataset", "question_id", "question", "context", "correct_answer", "context_used"]]


def load_co_sqa_questions(datasets_dir: Path, context_used: bool = False) -> DataFrame:
    pkl_path = datasets_dir / "CoSQA" / "json-train.parquet"
    if not pkl_path.exists():
        return empty_questions_df

    df = DataFrame(pd.read_parquet(pkl_path, filters=[("label", "==", 1)]))
    df["dataset"] = "CoSQA"
    df = df.reset_index(names="question_id").rename(columns={"doc": "question", "code": "correct_answer"})
    df["context_used"] = context_used
    df["context"] = (
        df["docstring_tokens"] if context_used else None
    )  # Не совсем верно. Скорее для этого датасета вообще нет контекста (тут всегда поиск подходящего)
    return df[["dataset", "question_id", "question", "context", "correct_answer", "context_used"]]


def load_simplified_nq_questions_raw(datasets_dir: Path, context_used: bool = False) -> pd.DataFrame:
    file_path = datasets_dir / "natural_questions" / "simplified-nq-train.jsonl.gz"
    if not file_path.exists():
        return empty_questions_df

    batch_size = 5000  # Количество строк в одной порции
    df_chunks = []
    records_buffer = []
    skipped_count = 0
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records_buffer.append(record)
                except json.JSONDecodeError:
                    skipped_count += 1
                    continue

                # Создание порции DataFrame при заполнении буфера
                if len(records_buffer) >= batch_size:
                    chunk_df = pd.DataFrame(records_buffer)
                    df_chunks.append(chunk_df)
                    records_buffer = []
                    gc.collect()  # Принудительная сборка мусора

            # Обработка остатка строк
            if records_buffer:
                chunk_df = pd.DataFrame(records_buffer)
                df_chunks.append(chunk_df)

    except MemoryError:
        logger.warning(f"Превышение памяти на этапе чтения. Загружено порций: {len(df_chunks)}")
    # Конкатенация всех порций
    for i, chunk in enumerate(df_chunks):
        chunk["extracted_short_answers"] = chunk.apply(
            lambda row: extract_answer_texts(row["annotations"], row["document_html"], "short_answers"), axis=1
        )

        chunk["extracted_long_answers"] = chunk.apply(
            lambda row: extract_answer_texts(row["annotations"], row["document_html"], "long_answer"), axis=1
        )
        chunk["context"] = parallel_clean(chunk, "document_html") if context_used else None
        # chunk[short_answers'] = chunk['annotations'].apply(lambda x: extract_annotation_info(x)[0])
        chunk["yes_no_answer"] = chunk["annotations"].apply(lambda x: extract_annotation_info(x)[1])
        chunk = chunk[
            chunk["extracted_short_answers"].isna()
            & chunk["extracted_long_answers"].isna()
            & chunk["yes_no_answer"].isna()
        ].drop("document_html")
        df_chunks[i] = chunk
    try:
        df = pd.concat(df_chunks, ignore_index=True)
        logger.info(f"Итоговый размер: {len(df)} строк {df.info()}. Пропущено ошибок: {skipped_count}")
    except MemoryError:
        logger.warning("Недостаточно памяти для объединения всех порций в один DataFrame. Берем первую")
        df = df_chunks[0]
    return df


def load_simplified_nq_questions(datasets_dir: Path, context_used: bool = False) -> pd.DataFrame:
    logger.info("Очень долго грузит, проще закомитить вопросы")
    saved_datasets = datasets_dir / f"nq_{context_used}.csv"
    if saved_datasets.exists():
        df = pd.read_csv(saved_datasets)
    else:
        df = load_simplified_nq_questions_raw(datasets_dir, context_used)

        df["dataset"] = "NQ"
        df["correct_answer"] = (
            df["yes_no_answer"].apply(lambda x: [] if x == "NONE" else [x])
            + df["extracted_long_answers"]
            + df["extracted_short_answers"]
        )
        df = df.reset_index(names="question_id").rename(columns={"question_text": "question"})
        df["context_used"] = context_used
        df[
            [
                "question_id",
                "document_title",
                "document_url",
                "question",
                "context",
                "context_used",
                "dataset",
                "correct_answer",
            ]
        ].to_csv(rf"D:\P_work\Rag-VKR_copy\datasets\nq_{context_used}.csv", index=False)
    return df[["dataset", "question_id", "question", "context", "correct_answer", "context_used"]]


def extract_annotation_info(annotations_list):
    """Извлекает тексты коротких ответов и yes/no метку из аннотаций."""
    if not annotations_list or not isinstance(annotations_list, list):
        return [], None

    short_answers = []
    yes_no = None

    for ann in annotations_list:
        # Yes/No ответ
        yes_no = ann.get("yes_no_answer", "NONE")

        # Короткие ответы (могут быть в виде токенов или готового текста)
        for sa in ann.get("short_answers", []):
            if "text" in sa and sa["text"]:
                short_answers.append(sa["text"])
            elif "start_token" in sa and "end_token" in sa:
                # Если только токены — отметим для последующей экстракции
                short_answers.append({"start_token": sa["start_token"], "end_token": sa["end_token"]})

    return short_answers, yes_no


def clean_html(html_string):
    """Преобразует HTML в чистый текст, пригодный для LLM."""
    if not isinstance(html_string, str):
        return ""

    soup = BeautifulSoup(html_string, "lxml")

    # Удаление скриптов, стилей, навигации
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Получение текста и нормализация пробелов
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)  # Замена множественных пробелов на один
    return text  # Ограничение длины при необходимости


def extract_text_by_bytes(html_string, start_byte, end_byte):
    """Извлекает подстроку из очищенного HTML по байтовым смещениям."""
    if not isinstance(html_string, str):
        return ""
    # Важно: после очистки HTML байтовые смещения могут не совпадать
    # Поэтому лучше извлекать из исходного HTML, затем чистить результат
    try:
        raw_span = html_string[start_byte:end_byte]
        return clean_html(raw_span)
    except:
        return ""


# Пример применения (если в аннотациях есть byte-офсеты)
def extract_answer_texts(annotations_list, document_html, ans_type="short_answers"):
    texts = []
    if not annotations_list or not isinstance(annotations_list, list):
        return texts
    for ann in annotations_list:
        ann = ann.get(ans_type, [])
        if isinstance(ann, dict):
            ann = [ann]
        for sa in ann:
            if "start_byte" in sa and "end_byte" in sa:
                span_text = extract_text_by_bytes(document_html, sa["start_byte"], sa["end_byte"])
                if span_text:
                    texts.append(span_text)
    return texts


def load_all_questions(project_root: Path, context_used: bool = False) -> DataFrame:
    """
    Получаем полностью все вопросы по всем датасетам
    Args:
        project_root: адрес где папка с датасетами

    Returns:
        list[dict[str, Any]]: список всех вопросов
    """
    datasets_dir = project_root / "datasets"
    saved_datasets = datasets_dir / f"all_q_{context_used}.csv"
    if saved_datasets.exists():
        loaded = pd.read_csv(saved_datasets)
        loaded = loaded[loaded["question_id"] < 50]  # todo ускорение расчетов Осталось: 91242
        return cast(DataFrame, loaded)
    all_q = load_ru_rag_questions(datasets_dir, context_used)
    if all_q is None:
        all_q = empty_questions_df
    all_q = pd.concat([all_q, load_co_sqa_questions(datasets_dir, context_used)], ignore_index=True)
    all_q = pd.concat([all_q, load_simplified_nq_questions(datasets_dir, context_used)], ignore_index=True)

    all_q.to_csv(saved_datasets, index=False)
    logger.info("Finish load")
    return all_q


# Параллельные вычисления

import re
from lxml import html as lxml_html

# Компилируем регулянку один раз
WHITESPACE_RE = re.compile(r"\s+")

# Теги для удаления
REMOVE_TAGS = {"script", "style", "nav", "footer", "header"}


def clean_html_fast(html_string):
    if not isinstance(html_string, str) or not html_string:
        return ""

    try:
        # lxml.html быстрее, чем BeautifulSoup
        doc = lxml_html.fromstring(html_string)

        # Удаление тегов через lxml (эффективнее)
        for tag in REMOVE_TAGS:
            for element in doc.xpath(f".//{tag}"):
                parent = element.getparent()
                if parent is not None:
                    parent.remove(element)

        # Получение текста
        text = doc.text_content()
        # Нормализация пробелов
        return WHITESPACE_RE.sub(" ", text).strip()
    except Exception:
        return ""


import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


# Функция должна быть определена на верхнем уровне (для pickle)
def process_chunk(args):
    html_list, indices = args
    results = []
    for h in html_list:
        results.append(clean_html_fast(h))  # Используем оптимизированную функцию из п.1
    return indices, results


def parallel_clean(df, column_name, max_workers=None):
    if max_workers is None:
        max_workers = mp.cpu_count()

    # Разбиваем данные на части
    html_data = df[column_name].tolist()
    indices = df.index.tolist()

    # Оптимальный размер чанка
    chunk_size = len(html_data) // max_workers + 1
    chunks = []

    for i in range(0, len(html_data), chunk_size):
        chunks.append((html_data[i : i + chunk_size], indices[i : i + chunk_size]))

    results = [None] * len(df)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx_chunk, res_chunk in executor.map(process_chunk, chunks):
            for i, res in zip(idx_chunk, res_chunk):
                results[i] = res

    return results


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    out_dir = project_root / "test_datasets"
    questions = load_all_questions(project_root, context_used=True)
