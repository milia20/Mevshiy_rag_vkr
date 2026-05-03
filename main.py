"""Главный файл в котором я вызываю все функции"""

import os
import sys

from src import extract_and_download, load_setup_settings
from src.logger import logger


def main() -> None:
    """
    все функции по порядку, если TEST=true, то делает 1 запрос для llm и только 2 первые модели
    """
    try:
        logger.info("Скачиваем датасеты [TEST MODE]")
        settings = load_setup_settings()
        extract_and_download(settings)
    except Exception as e:
        logger.error(f"Необработанная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Начало общей функции", flush=True)
    if os.getenv("TEST_MOCK_MAIN") == "1":
        print("[MOCK]")
    else:
        main()
    sys.exit(0)
