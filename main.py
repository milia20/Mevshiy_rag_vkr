"""Главный файл в котором я вызываю все функции"""

import os
import sys

from dataset_retrieval_runner import main_run
from src import extract_and_download, load_setup_settings
from src.logger import logger
from src.test_datasets_runner import run_test_datasets


def main() -> None:
    """
    все функции по порядку, если TEST=true, то делает 1 запрос для llm и только 2 первые модели
    """
    try:
        settings = load_setup_settings()
        logger.info(f"Скачиваем датасеты [TEST MODE] = {settings.test}")
        extract_and_download(settings)
        logger.info(f"считаем ответы [TEST MODE] = {settings.test}")
        run_test_datasets(settings)
        logger.info(f"считаем ответы [TEST MODE] = {settings.test}")
        main_run(settings)
    except Exception as e:
        logger.error(f"Необработанная ошибка: {e}")
        raise e
        sys.exit(1)


if __name__ == "__main__":
    print("Начало общей функции", flush=True)
    if os.getenv("TEST_MOCK_MAIN") == "1":
        print("[MOCK]")
    else:
        main()
    # import pandas as pd
    #
    # splits = {'train': 'cosqa-train.json', 'validation': 'cosqa-dev.json'}
    # df = pd.read_json("hf://datasets/gonglinyuan/CoSQA/" + splits["train"])
    sys.exit(0)
