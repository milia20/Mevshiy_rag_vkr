import json
import os
import pickle
import random
import re
import time
from typing import List, Dict, Any

import pandas as pd
import requests
from tqdm import tqdm

# Configuration
# OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Update with your Ollama API URL
OLLAMA_API_URL = "http://127.0.0.1:1234/v1/completions"  # Update with your Ollama API URL
MODEL_NAME = "google/gemma-3-12b"  # or any other model you want to use
OUTPUT_CSV = "evaluation_results.csv"
sample_size = 1_000_000_000


class OllamaClient:
    def __init__(self, model_name: str = MODEL_NAME, api_url: str = OLLAMA_API_URL):
        self.model_name = model_name
        self.api_url = api_url

    def generate_answer(self, question: str, max_retries: int = 3) -> str:
        """Generate an answer to the given question using Ollama API."""
        prompt = f"""Answer the following question concisely. If you don't know the answer, say 'I don't know'.

Question: {question}
Answer:"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json().get("choices")[0].get("text", "I don't know").strip()
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt == max_retries - 1:
                    print(f"Error generating answer after {max_retries} attempts: {e}")
                    return "I don't know"
                time.sleep(2 ** attempt)  # Exponential backoff


def load_ru_rag_dataset(filepath: str) -> pd.DataFrame:
    """Load the Russian RAG test dataset."""
    with open(filepath, 'rb') as f:
        df = pickle.load(f)
    return df


def evaluate_ru_rag(ollama_client: OllamaClient, df: pd.DataFrame, sample_size: int = 50) -> List[
    Dict[str, Any]]:
    """Evaluate the model on the Russian RAG test dataset."""
    results = []

    # Sample a subset if the dataset is large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Russian RAG dataset"):
        question = row['Вопрос']
        correct_answer = row['Правильный ответ']

        # Generate answer without context
        generated_answer = ollama_client.generate_answer(question)

        # Simple exact match evaluation (you can implement more sophisticated evaluation)
        is_correct = generated_answer.lower() in correct_answer.lower() or correct_answer.lower() in generated_answer.lower()

        results.append({
            'dataset': 'ru_rag',
            'question': question,
            'correct_answer': correct_answer,
            'generated_answer': generated_answer,
            'is_correct': is_correct,
            'context_used': False
        })

        # Be nice to the API
        time.sleep(1)

    return results


def evaluate_natural_questions(ollama_client: OllamaClient, nq_test_file: str,
                               sample_size: int = 50) -> List[Dict[str, Any]]:
    """Evaluate the model on the Natural Questions test dataset."""
    results = []

    # Load Natural Questions test data
    with open(nq_test_file, 'r', encoding='utf-8') as f:
        nq_data = [json.loads(line) for line in f]

    # Sample a subset if the dataset is large
    if len(nq_data) > sample_size:
        nq_data = random.sample(nq_data, sample_size)

    for item in tqdm(nq_data, desc="Evaluating Natural Questions"):
        question = item['question_text']

        # Get the first short answer as the correct answer (simplified)
        correct_answers = []
        for annotation in item['annotations']:
            for short_answer in annotation.get('short_answers', []):
                if short_answer.get('text'):
                    correct_answers.append(short_answer['text'])

        if not correct_answers:
            continue  # Skip if no short answers

        correct_answer = correct_answers[0]  # Take the first answer for simplicity

        # Generate answer without context
        generated_answer = ollama_client.generate_answer(question)

        # Complex evaluation using multiple methods
        is_correct = evaluate_answer_complex(generated_answer, correct_answer)

        results.append({
            'dataset': 'natural_questions',
            'question': question,
            'correct_answer': correct_answer,
            'generated_answer': generated_answer,
            'is_correct': is_correct,
            'context_used': False
        })

        # Be nice to the API
        time.sleep(1)

    return results


def evaluate_answer_complex(generated_answer: str, correct_answer: str) -> bool:
    """
    Evaluate if the generated answer matches the correct answer using multiple evaluation methods.
    Returns True if the answer is considered correct based on any of the evaluation methods.
    """
    # Normalize both answers
    gen_norm = normalize_text(generated_answer)
    corr_norm = normalize_text(correct_answer)

    # Method 1: Exact match after normalization
    if gen_norm == corr_norm:
        return True # 100%

    # Method 2: Generated answer contains the correct answer or vice versa (case-insensitive)
    if gen_norm in corr_norm or corr_norm in gen_norm:
        return True # (abs(len(corr_norm) - len(gen_norm)) // len(corr_norm) ) / len(corr_norm)

    # Method 3: Token overlap - check if most tokens from correct answer are in generated answer
    gen_tokens = set(gen_norm.split())
    corr_tokens = set(corr_norm.split())

    if corr_tokens and len(gen_tokens.intersection(corr_tokens)) / len(corr_tokens) >= 0.8:
        return True

    # Method 4: Check for numeric answers - if both contain numbers, compare them
    gen_nums = extract_numbers(generated_answer)
    corr_nums = extract_numbers(correct_answer)

    if gen_nums and corr_nums:
        # If both have numbers, check if they match
        if set(gen_nums) == set(corr_nums):
            return True

    # Method 5: Fuzzy matching using sequence similarity
    if calculate_similarity(gen_norm, corr_norm) >= 0.85:
        return True

    # Method 6: Check if generated answer contains the correct answer with additional context
    if is_substring_with_flexibility(gen_norm, corr_norm):
        return True

    return False


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase, removing extra spaces, and common punctuation."""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove common punctuation while preserving word boundaries
    text = re.sub(r'[^\w\s]', ' ', text)
    # Clean up extra spaces after punctuation removal
    text = ' '.join(text.split())
    return text


def extract_numbers(text: str) -> List[str]:
    """Extract all numbers (integers and floats) from text."""
    # Pattern to match integers and floating point numbers
    number_pattern = r'\d+\.?\d*'
    numbers = re.findall(number_pattern, text)
    # Filter out empty strings and convert to a cleaned list
    return [num for num in numbers if num]


def calculate_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings using a simple ratio of matching characters."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Use a simple character-based similarity
    matches = 0
    max_len = max(len(s1), len(s2))

    # Pad the shorter string with spaces for comparison
    s1_padded = s1.ljust(max_len)
    s2_padded = s2.ljust(max_len)

    for c1, c2 in zip(s1_padded, s2_padded):
        if c1 == c2:
            matches += 1

    return matches / max_len


def is_substring_with_flexibility(needle: str, haystack: str) -> bool:
    """
    Check if needle is in haystack with some flexibility for surrounding words.
    This handles cases where the correct answer is in the generated answer but with
    additional context or slightly different phrasing.
    """
    if not needle or not haystack:
        return False

    # If the needle is very short, require exact match
    if len(needle) < 3:
        return needle == haystack

    # Check if the shorter string is contained in the longer one
    if len(needle) <= len(haystack):
        return is_contained_with_tolerance(needle, haystack)
    else:
        return is_contained_with_tolerance(haystack, needle)


def is_contained_with_tolerance(needle: str, haystack: str) -> bool:
    """Check if needle is contained in haystack with some tolerance for word boundaries."""
    # Split both into words
    needle_words = set(needle.split())
    haystack_words = set(haystack.split())

    # If needle has 1-2 words, check if all words are in haystack
    if len(needle_words) <= 2:
        return needle_words.issubset(haystack_words)

    # For longer needles, check if majority of words are present
    intersection = needle_words.intersection(haystack_words)
    if len(needle_words) == 0:
        return True  # Empty needle is always contained
    return len(intersection) / len(needle_words) >= 0.7


def main():
    # Initialize Ollama client
    ollama_client = OllamaClient()

    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
    except requests.RequestException as e:
        print("Error: Could not connect to Ollama API. Please make sure Ollama is running.")
        print(f"Error details: {e}")
        return

    all_results = []

    # Evaluate Russian RAG dataset if available
    ru_rag_path = os.path.join("ru_rag_test_dataset-main", "ru_rag_test_dataset.pkl")
    if os.path.exists(ru_rag_path):
        print("Loading Russian RAG dataset...")
        df_ru_rag = load_ru_rag_dataset(ru_rag_path)
        print(
            f"Evaluating on {min(sample_size, len(df_ru_rag))} samples from Russian RAG dataset...")
        ru_rag_results = evaluate_ru_rag(ollama_client, df_ru_rag, sample_size=sample_size)
        all_results.extend(ru_rag_results)

    # Evaluate Natural Questions dataset if available
    nq_test_path = os.path.join("natural-questions-master",
                                "nq-test-sample.jsonl")  # Update with actual path
    if os.path.exists(nq_test_path):
        print(f"Evaluating on Natural Questions dataset from {nq_test_path}...")
        nq_results = evaluate_natural_questions(ollama_client, nq_test_path,
                                                sample_size=sample_size)
        all_results.extend(nq_results)

    # Save results to CSV
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"\nEvaluation complete! Results saved to {OUTPUT_CSV}")

        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Total questions evaluated: {len(all_results)}")
        print(f"Correct answers: {df_results['is_correct'].sum()}")
        print(f"Accuracy: {df_results['is_correct'].mean():.2f}")

        # Print some example results
        print("\n=== Example Results ===")
        for i, row in df_results.sample(min(3, len(df_results))).iterrows():
            print(f"\nQuestion: {row['question']}")
            print(f"Correct answer: {row['correct_answer']}")
            print(f"Generated answer: {row['generated_answer']}")
            print(f"Correct: {row['is_correct']}")
    else:
        print("No results to save. Please check if the dataset paths are correct.")


if __name__ == "__main__":
    main()
