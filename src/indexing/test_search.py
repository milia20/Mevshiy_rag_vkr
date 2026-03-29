import logging
import sys

from src.indexing.qdrant_uploader import QdrantIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "test_hnsw_default"   # или любая ваша коллекция, где есть данные
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # та же модель, что при индексации

def main():
    indexer = QdrantIndexer(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        in_memory=False,      # используем постоянное хранилище
    )

    if not indexer.client.collection_exists(COLLECTION_NAME):
        print(f"Ошибка: коллекция '{COLLECTION_NAME}' не найдена в Qdrant.")
        print("Убедитесь, что вы сначала запустили скрипт индексации и загрузили данные.")
        sys.exit(1)

    question = "Когда родился Петр 1?"
    question = "app.websockets params"

    print(f"\nПоиск по вопросу: {question}\n")

    try:
        results = indexer.search(
            query_text=question,
            limit=3, # сколько лучших фрагментов показать
            model_name=EMBEDDING_MODEL,
            collection_name=COLLECTION_NAME,
            score_threshold=0.5,         # отсекаем совсем непохожие
        )
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        sys.exit(1)

    if not results:
        print("Ничего не найдено. Возможно, в коллекции нет информации о Петре I.")
    else:
        print(f"Найдено {len(results)} фрагментов:\n")
        for i, res in enumerate(results, 1):
            print(f"--- Результат {i} (score = {res['score']:.4f}) ---")
            text = res['payload'].get('text', '')
            print(f"Текст: {text[:500]}{'...' if len(text) > 500 else ''}")
            source = res['payload'].get('source', 'неизвестно')
            doc_title = res['payload'].get('doc_title', '')
            if doc_title:
                print(f"Источник: {doc_title} ({source})")
            elif source != 'неизвестно':
                print(f"Источник: {source}")
            print()

    indexer.close()

if __name__ == "__main__":
    main()