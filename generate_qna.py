import json
import os
import re
from pathlib import Path

# OpenRouter client for generating Q&A pairs
class OpenRouterClient:
    def __init__(self, model_name: str = "anthropic/claude-3-haiku", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")

    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        import requests
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                return response.json().get("choices")[0].get("message", {}).get("content", "").strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error generating answer after {max_retries} attempts: {e}")
                    return ""
                import time
                time.sleep(2 ** attempt)

# Legacy Ollama client for backward compatibility
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

# Generate Q&A pairs for a language using OpenRouter
def generate_qna_for_lang(lang_dir: Path, client, max_pairs: int = 200):
    """Generate Q&A pairs for the given language directory using OpenRouter."""
    pairs = []
    md_file = next(lang_dir.rglob("*.md"), None)
    
    if not md_file:
        print(f"No markdown files found in {lang_dir}")
        return pairs
    
    # Read the markdown content
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {md_file}: {e}")
        return pairs
    
    file_path = str(md_file.relative_to(Path("fastapi_doc")))
    
    # Generate Q&A pairs in batches
    batch_size = 5  # Generate 5 pairs at a time
    for batch_start in range(0, max_pairs, batch_size):
        current_batch_size = min(batch_size, max_pairs - batch_start)
        
        prompt = f"""
Based on the following documentation content, generate {current_batch_size} question-answer pairs.
The questions should be practical and relevant to developers using this documentation.
Return the response as a JSON array of objects, each with "question" and "answer" fields.

Documentation content:
{content[:3000]}  # Limit content to avoid token limits

Example format:
[
    {{"question": "How do I install FastAPI?", "answer": "You can install FastAPI using pip: pip install fastapi"}},
    {{"question": "What is dependency injection in FastAPI?", "answer": "Dependency injection is a way to declare dependencies for your path operations..."}}
]

Generate {current_batch_size} diverse and useful Q&A pairs:
"""
        
        try:
            response = client.generate_answer(prompt)
            if response:
                batch_pairs = parse_qna(response)
                for question, answer in batch_pairs:
                    if question and answer:
                        pairs.append({
                            "question": question,
                            "answer": answer,
                            "file": file_path,
                        })
                
                print(f"Generated {len(batch_pairs)} pairs for {file_path} (batch {batch_start//batch_size + 1})")
                
                if len(pairs) >= max_pairs:
                    break
            else:
                print(f"No response generated for {file_path}")
        except Exception as e:
            print(f"Error generating Q&A for {file_path}: {e}")
    
    return pairs[:max_pairs]  # Ensure we don't exceed max_pairs

if __name__ == "__main__":
    model_name = os.getenv("OPENROUTER_MODEL", "anthropic/c")
    
    try:
        # Initialize OpenRouter client
        client = OpenRouterClient(model_name=model_name)
        print(f"Using OpenRouter model: {model_name}")
        
        base = Path("fastapi_doc")
        en_dir = base / "en" / "docs"
        ru_dir = base / "ru" / "docs"
        
        # Generate Q&A pairs for each language
        print("Generating Q&A pairs for English documentation...")
        en_pairs = generate_qna_for_lang(en_dir, client, 200)
        
        print("Generating Q&A pairs for Russian documentation...")
        ru_pairs = generate_qna_for_lang(ru_dir, client, 200)
        
        all_pairs = en_pairs + ru_pairs
        out_path = Path("qna_pairs.jsonl")
        
        with out_path.open("w", encoding="utf-8") as f:
            for item in all_pairs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"Generated {len(all_pairs)}. OutputЖ {out_path}")
        print(f"English: {len(en_pairs)}, Russian: {len(ru_pairs)}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set the OPENROUTER_API_KEY environment variable or pass it to the OpenRouterClient constructor.")
    except Exception as e:
        print(f"Error during execution: {e}")
# set OPENROUTER_API_KEY=