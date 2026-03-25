import json
import pickle
import uuid
from pathlib import Path

from mk_plugin.document_processor import clean_markdown, create_text_splitter


def transform_dataset(
    dataset_path: str,
    output_qa_path: str,
    output_chunks_path: str,
    files_dir: str = None,               # добавлен параметр
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """
    Преобразует датасет ru_rag_test_dataset в два файла:
        - QA пары (вопрос-ответ)
        - Чанки контекстов с метаданными для индексации.
        Текст для чанков берётся из файлов, указанных в столбце 'Файл'.

    Parameters
    ----------
    dataset_path : str
        Путь к файлу .pkl с датафреймом.
    output_qa_path : str
        Путь для сохранения JSONL с QA парами.
    output_chunks_path : str
        Путь для сохранения JSONL с чанками.
    files_dir : str
        Путь к папке с файлами. Если None, то используется папка 'files'
        в той же директории, что и dataset_path.
    chunk_size : int
        Размер чанка.
    chunk_overlap : int
        Перекрытие между чанками.
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        df = pickle.load(f)

    # Убедимся, что столбцы есть
    required_columns = ['Вопрос', 'Правильный ответ', 'Контекст', 'Файл']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe. Available: {df.columns.tolist()}")

    # Определяем папку с файлами
    if files_dir is None:
        files_dir = Path(dataset_path).parent / "files"
    else:
        files_dir = Path(files_dir)

    if not files_dir.exists():
        raise FileNotFoundError(f"Files directory not found: {files_dir}")

    splitter = create_text_splitter(chunk_size, chunk_overlap)

    Path(output_qa_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_chunks_path).parent.mkdir(parents=True, exist_ok=True)

    qa_pairs = []
    all_chunks = []

    for idx, row in df.iterrows():
        question = row['Вопрос']
        answer = row['Правильный ответ']
        source_file = row['Файл']
        context_column = row['Контекст']   # сохраняем для метаданных

        # Добавляем QA пару
        qa_pairs.append({
            "id": idx,
            "question": question,
            "answer": answer
        })

        # Читаем содержимое файла
        file_path = files_dir / source_file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Skipping chunk generation for row {idx}.")
            continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}. Skipping.")
            continue

        # Очищаем содержимое от маркдауна
        cleaned_content = clean_markdown(raw_content)

        # Разбиваем на чанки
        chunks = splitter.split_text(cleaned_content)

        # Для каждого чанка создаём запись с метаданными
        for chunk_text in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_record = {
                "text": chunk_text,
                "metadata": {
                    "source": source_file,
                    "question": question,
                    "answer": answer,
                    "context_from_column": context_column,   # опционально
                    "chunk_id": chunk_id,
                    "row_index": idx
                }
            }
            all_chunks.append(chunk_record)

    print(f"Saving {len(qa_pairs)} QA pairs to {output_qa_path}...")
    with open(output_qa_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Saving {len(all_chunks)} chunks to {output_chunks_path}...")
    with open(output_chunks_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print("Done")


if __name__ == "__main__":
    DATASET_PATH = r"D:\P_work\Rag-VKR\ru_rag_test_dataset-main\ru_rag_test_dataset.pkl"
    QA_OUTPUT = r"data/qa_pairs.jsonl"
    CHUNKS_OUTPUT = r"data/chunks.jsonl"
    FILES_DIR = r"D:\P_work\Rag-VKR\ru_rag_test_dataset-main\files"  # None для автоопределения

    transform_dataset(
        dataset_path=DATASET_PATH,
        output_qa_path=QA_OUTPUT,
        output_chunks_path=CHUNKS_OUTPUT,
        files_dir=FILES_DIR,
        chunk_size=512,
        chunk_overlap=50,
    )