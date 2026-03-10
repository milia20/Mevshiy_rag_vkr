import json
import os
import re
from pathlib import Path

# Ollama client similar to generate_and_eval.py
class OllamaClient:
    def __init__(self, model_name: str = "google/gemma-3-12b", api_url: str = "http://127.0.0.1:1234/v1/completions"):
        self.model_name = model_name
        self.api_url = api_url

    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        import requests
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json().get("choices")[0].get("text", "").strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error generating answer after {max_retries} attempts: {e}")
                    return ""
                import time
                time.sleep(2 ** attempt)

# Helper to extract Q&A pairs from model output

def parse_qna(json_text: str):
    try:
        data = json.loads(json_text)
        if isinstance(data, list):
            return [(item.get("question"), item.get("answer")) for item in data]
    except Exception:
        pass
    return []

# Generate Q&A pairs for a language

def generate_qna_for_lang(lang_dir: Path, max_pairs: int = 200):
    """Generate placeholder Q&A pairs for the given language directory."""
    pairs = []
    md_file = next(lang_dir.rglob("*.md"), None)
    file_path = str(md_file.relative_to(Path("fastapi_doc"))) if md_file else "unknown.md"
    for idx in range(1, max_pairs + 1):
        pairs.append({
            "question": f"Placeholder question {idx} for {file_path}",
            "answer": f"Placeholder answer {idx} for {file_path}",
            "file": file_path,
        })
    return pairs

if __name__ == "__main__":
    base = Path("fastapi_doc")
    en_dir = base / "en" / "docs"
    ru_dir = base / "ru" / "docs"
    en_pairs = generate_qna_for_lang(en_dir, 200)
    ru_pairs = generate_qna_for_lang(ru_dir, 200)
    all_pairs = en_pairs + ru_pairs
    out_path = Path("qna_pairs.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for item in all_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(all_pairs)} Q&A pairs. Output written to {out_path}")
