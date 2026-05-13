from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from llm_generate.llm import _build_model_configs, generate_answer
from load_dataets import load_all_questions
from src.config import SetupSettings
from src.logger import logger


def run_test_datasets(settings: SetupSettings) -> None:
    """
    Главная функция которая считает все датасеты
    :param settings: Настройки системы
    :return: Просто вывод
    """
    project_root = Path(__file__).parent.parent
    out_dir = project_root / "test_datasets"
    out_csv = out_dir / "evaluation_results_all.csv"

    encoding = os.getenv("CSV_ENCODING", "mbcs")

    test_mode = bool(getattr(settings, "test", False))
    # test_mode = False
    questions = load_all_questions(project_root)
    # questions = df_to_list_of_dicts(questions)
    if test_mode:
        questions = questions[:2]

    if questions.empty:
        logger.warning("No questions found for evaluation")
        return

    model_configs = _build_model_configs(settings, test_mode=test_mode)
    if not model_configs:
        logger.warning("No LLMs configured")
        return

    models_str = [str(mod) for mod in model_configs]
    questions["model"] = [models_str] * len(questions)
    questions = questions.explode("model")

    if out_csv.exists() and out_csv.stat().st_size > 0:
        df_done = pd.read_csv(out_csv, usecols=["dataset", "question_id", "model"])
        questions = pd.merge(questions, df_done, how="left", on=["dataset", "question_id", "model"], indicator=True)
        questions = questions[questions["_merge"] != "both"].drop("_merge", axis=1)
    questions = questions.sort_values(["model", "dataset"], ascending=False)
    if not questions.empty:
        logger.info(f"Осталось: {len(questions)}")
        model_configs: dict = {str(mod): mod for mod in model_configs}
        disable_tqdm = len(questions) == 0
        for _, q in tqdm(questions.iterrows(), desc="Evaluating datasets", file=sys.stdout, disable=disable_tqdm):
            key = model_configs[q["model"]]
            if q["dataset"] == "CoSQA":
                q["question"] = "Write the function code as an answer. Question:" + q["question"]
                q["question_id"] += 100_000
                generate_answer(key, q, out_csv, encoding)
            else:
                generate_answer(key, q, out_csv, encoding)

    else:
        logger.info("Уже все посчитано")
        # out_csv.unlink(missing_ok=True)
