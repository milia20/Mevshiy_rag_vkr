import os
import random
import re
import time
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# Configuration
OLLAMA_API_URL = "http://127.0.0.1:1234/v1/completions"
MODEL_NAME = "google/gemma-3-12b"
OUTPUT_CSV = "codeqa_evaluation_code_results.csv"
OUTPUT_CSV_SIMPLE = "codeqa_evaluation_code_results_simplified.csv"
DEFAULT_TEST_SAMPLE_SIZE = 1000


class OllamaClient:
    def __init__(self, model_name: str = MODEL_NAME, api_url: str = OLLAMA_API_URL):
        self.model_name = model_name
        self.api_url = api_url

    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json().get("choices")[0].get("text", "I don't know").strip()
            except (requests.RequestException, ValueError) as e:
                if attempt == max_retries - 1:
                    return "I don't know"
                time.sleep(2 ** attempt)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = " ".join(text.split())
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


def extract_numbers(text: str) -> List[str]:
    number_pattern = r"\d+\.?\d*"
    numbers = re.findall(number_pattern, text)
    return [num for num in numbers if num]


def calculate_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    matches = 0
    max_len = max(len(s1), len(s2))
    s1_padded = s1.ljust(max_len)
    s2_padded = s2.ljust(max_len)
    for c1, c2 in zip(s1_padded, s2_padded):
        if c1 == c2:
            matches += 1
    return matches / max_len


def is_contained_with_tolerance(needle: str, haystack: str) -> bool:
    needle_words = set(needle.split())
    haystack_words = set(haystack.split())
    if len(needle_words) <= 2:
        return needle_words.issubset(haystack_words)
    intersection = needle_words.intersection(haystack_words)
    if len(needle_words) == 0:
        return True
    return len(intersection) / len(needle_words) >= 0.7


def is_substring_with_flexibility(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    if len(needle) < 3:
        return needle == haystack
    if len(needle) <= len(haystack):
        return is_contained_with_tolerance(needle, haystack)
    else:
        return is_contained_with_tolerance(haystack, needle)


def evaluate_answer_complex(generated_answer: str, correct_answer: str) -> bool:
    gen_norm = normalize_text(generated_answer)
    corr_norm = normalize_text(correct_answer)
    if gen_norm == corr_norm:
        return True
    if gen_norm in corr_norm or corr_norm in gen_norm:
        return True
    gen_tokens = set(gen_norm.split())
    corr_tokens = set(corr_norm.split())
    if corr_tokens and len(gen_tokens.intersection(corr_tokens)) / len(corr_tokens) >= 0.8:
        return True
    gen_nums = extract_numbers(generated_answer)
    corr_nums = extract_numbers(correct_answer)
    if gen_nums and corr_nums and set(gen_nums) == set(corr_nums):
        return True
    if calculate_similarity(gen_norm, corr_norm) >= 0.85:
        return True
    if is_substring_with_flexibility(gen_norm, corr_norm):
        return True
    return False


def build_prompt(code: str, question: str) -> str:
    return (
        "You are given a Python code snippet and a question about it. "
        "Answer concisely. If you don't know, say 'I don't know'.\n\n"
        "Code:\n" + code + "\n\n"
        "Question: " + question + "\n"
        "Answer:"
    )


def read_lines(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_triplets(question_path: str, code_path: str, answer_path: str) -> List[Tuple[str, str, str]]:
    qs = read_lines(question_path) if os.path.exists(question_path) else []
    cs = read_lines(code_path) if os.path.exists(code_path) else []
    ans = read_lines(answer_path) if os.path.exists(answer_path) else []
    if len(qs) == 0 or len(cs) == 0 or len(ans) == 0:
        return []
    n = min(len(qs), len(cs), len(ans))
    return list(zip(qs[:n], cs[:n], ans[:n]))


def evaluate_codeqa(
    ollama_client: OllamaClient,
    dataset_name: str,
    question_path: str,
    code_path: str,
    answer_path: str,
    sample_size: int = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    triplets = load_triplets(question_path, code_path, answer_path)
    if not triplets:
        return [], []
    if sample_size and len(triplets) > sample_size:
        triplets = random.sample(triplets, sample_size)

    results_full: List[Dict[str, Any]] = []
    results_simple: List[Dict[str, Any]] = []

    for question, code, correct_answer in tqdm(triplets, desc=f"Evaluating {dataset_name}"):
        prompt = build_prompt(code, question)
        generated_answer = ollama_client.generate_answer(prompt)
        is_correct = evaluate_answer_complex(generated_answer, correct_answer)

        results_full.append(
            {
                "dataset": dataset_name,
                "question": question,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct,
                "context_used": True,
            }
        )
        results_simple.append(
            {
                "dataset": dataset_name,
                "question": question,
                "correct_answers": str([correct_answer]),
                "generated_answer": generated_answer,
                "is_correct": is_correct,
                "context_used": True,
            }
        )
        time.sleep(1)

    return results_full, results_simple


def main():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException:
        print("Warning: Could not connect to Ollama tags endpoint. Ensure your local API is running.")

    ollama_client = OllamaClient()

    all_full: List[Dict[str, Any]] = []
    all_simple: List[Dict[str, Any]] = []

    # data_sample/python
    ds_base = os.path.join("CodeQA-main", "data_sample", "python")
    ds_q = os.path.join(ds_base, "python_sample.question")
    ds_c = os.path.join(ds_base, "python_sample.code")
    ds_a = os.path.join(ds_base, "python_sample.answer")
    if os.path.exists(ds_q) and os.path.exists(ds_c) and os.path.exists(ds_a):
        f1, s1 = evaluate_codeqa(
            ollama_client,
            dataset_name="codeqa_python_sample",
            question_path=ds_q,
            code_path=ds_c,
            answer_path=ds_a,
            sample_size=None,
        )
        all_full.extend(f1)
        all_simple.extend(s1)

    # codeBERT/data/python/test
    test_base = os.path.join("CodeQA-main", "codeBERT", "data", "python", "test")
    tq = os.path.join(test_base, "test.question")
    tc = os.path.join(test_base, "test.code")
    ta = os.path.join(test_base, "test.answer")
    if os.path.exists(tq) and os.path.exists(tc) and os.path.exists(ta):
        f2, s2 = evaluate_codeqa(
            ollama_client,
            dataset_name="codeqa_python_test",
            question_path=tq,
            code_path=tc,
            answer_path=ta,
            sample_size=DEFAULT_TEST_SAMPLE_SIZE,
        )
        all_full.extend(f2)
        all_simple.extend(s2)

    if all_full:
        pd.DataFrame(all_full).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        pd.DataFrame(all_simple).to_csv(OUTPUT_CSV_SIMPLE, index=False, encoding="utf-8")
        print(f"Saved results to {OUTPUT_CSV} and {OUTPUT_CSV_SIMPLE}")
        print(f"Total evaluated: {len(all_full)} | Accuracy: {pd.DataFrame(all_full)['is_correct'].mean():.2f}")
    else:
        print("No datasets found. Check paths under CodeQA-main.")


if __name__ == "__main__":
    main()
