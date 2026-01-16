import json
import os
import random
import re
import time
from typing import List, Dict, Any

import requests
from tqdm import tqdm

# Configuration
OLLAMA_API_URL = "http://127.0.0.1:1234/v1/completions"  # Update with your Ollama API URL
MODEL_NAME = "google/gemma-3-12b"  # or any other model you want to use
OUTPUT_CSV = "evaluation_results_simplified.csv"
SAMPLE_SIZE = 100_000  # Adjust based on your needs


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
            "stream": False,
            "max_tokens": 100
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json().get("choices")[0].get("text", "I don't know").strip()
            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                if attempt == max_retries - 1:
                    print(f"Error generating answer after {max_retries} attempts: {e}")
                    return "I don't know"
                time.sleep(2 ** attempt)  # Exponential backoff


def load_simplified_nq_data(filepath: str, sample_size: int) -> List[Dict[str, Any]]:
    """
    Load and sample the simplified Natural Questions dataset with error handling.
    
    Args:
        filepath: Path to the simplified NQ JSONL file
        sample_size: Maximum number of samples to return
        
    Returns:
        List of valid data points with questions and answers
    """
    data = []
    error_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                # Validate the required fields
                if not isinstance(item, dict):
                    print(f"Warning: Line {i} is not a JSON object, skipping")
                    error_count += 1
                    continue
                    
                if 'question_text' not in item or 'annotations' not in item:
                    print(f"Warning: Line {i} is missing required fields, skipping")
                    error_count += 1
                    continue
                    
                # # Ensure answer is a list for consistent processing
                # if not isinstance(item['answer'], list):
                #     item['answer'] = [item['answer']]
                #
                # # Filter out empty answers
                # item['answer'] = [a for a in item['answer'] if a and str(a).strip()]
                #
                # if not item['answer']:
                #     continue
                #
                data.append(item)
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error on line {i}: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"Warning: Unexpected error on line {i}: {e}")
                error_count += 1
                continue
    
    if error_count > 0:
        print(f"Encountered {error_count} errors while loading the dataset")
    
    if not data:
        raise ValueError("No valid data found in the input file")
    
    # Sample if needed
    if len(data) > sample_size:
        data = random.sample(data, sample_size)
    
    print(f"Successfully loaded {len(data)} valid examples")
    return data


def extract_short_answers(item: Dict) -> List[str]:
    """Extract short answers from annotations."""
    annotations = item.get('annotations', [])
    answers = []
    for annotation in annotations:
        for short_answer in annotation.get('short_answers', []):
            end = short_answer['end_byte']
            star = short_answer['start_byte']
            text = ""
            start = False
            for t in item["document_tokens"]:
                if t["start_byte"] == star:
                    start = True

                if start and not t["html_token"]:
                    text += t["token"] + " "

                if t["end_byte"] == end:
                    break
            answers.append(text.strip())
    return answers

def evaluate_simplified_nq(ollama_client: OllamaClient, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate the model on the Natural Questions dataset."""
    results = []

    for item in tqdm(data, desc="Evaluating NQ"):
        question = item['question_text']

        # Extract all possible correct answers from annotations
        correct_answers = extract_short_answers(item)

        if not correct_answers:
            # Skip if no valid answers found
            continue

        # Generate answer without context
        generated_answer = ollama_client.generate_answer(question)

        # Evaluate the generated answer against all correct answers
        is_correct = any(
            evaluate_answer_complex(generated_answer, correct_answer)
            for correct_answer in correct_answers
        )

        results.append({
            'dataset': 'natural_questions',
            'question': question,
            'correct_answers': correct_answers,
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

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save evaluation results to a CSV file."""
    import pandas as pd
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Results saved to {output_file}")


def main():
    # Initialize the Ollama client
    ollama_client = OllamaClient()
    
    # Path to the simplified NQ dataset
    simplified_nq_file = "datasets/v1.0-simplified_nq-dev-all.jsonl/v1.0-simplified_nq-dev-all.jsonl"
    
    if not os.path.exists(simplified_nq_file):
        print(f"Error: {simplified_nq_file} not found. Please ensure the file exists in the current directory.")
        return
    
    # Load and evaluate the simplified NQ dataset
    print(f"Loading and sampling {SAMPLE_SIZE} examples from {simplified_nq_file}...")
    simplified_nq_data = load_simplified_nq_data(simplified_nq_file, SAMPLE_SIZE)
    
    print(f"Evaluating on {len(simplified_nq_data)} examples...")
    results = evaluate_simplified_nq(ollama_client, simplified_nq_data)
    
    # Calculate and print accuracy
    accuracy = sum(r['is_correct'] for r in results) / len(results) * 100
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2f}%")
    
    # Save results
    save_results(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()
