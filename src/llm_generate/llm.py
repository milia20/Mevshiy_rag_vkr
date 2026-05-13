from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import lmstudio as lms
import pandas as pd
import requests
from lmstudio import BaseModel
from lmstudio import LlmLoadModelConfigDict, LlmPredictionConfigDict, PredictionResult

from src.config import SetupSettings
from src.logger import logger


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    model: str
    thinking: bool
    temperature: float
    maxTokens: int


# SDK сам построит response_format, если передать lms.BaseModel
class OutputSchema(BaseModel):
    answer: str
    confidence: float


def _generate_structured_answer(
    backend: str,
    model: str,
    question: str,
    context: str | None,
    thinking: bool,
    temperature: float,
    max_tokens: int,
    answers: str | list,
) -> tuple[dict[str, Any], str]:
    prompt = build_structured_params(question=question, context=context, thinking=thinking)
    if context:
        question += f"\nContext:\n{context}"
    repeatPenalty = 1.1
    # Проблемная модель
    if model == "google/gemma-4-26b-a4b":
        repeatPenalty = 1.25

    if backend == "lmstudio_sdk":

        if llm := [loaded_model for loaded_model in lms.list_loaded_models("llm") if loaded_model.identifier == model]:
            llm = llm[0]
        else:
            for load_model in lms.list_loaded_models("llm"):
                load_model.unload()
            llm = lms.llm(model, ttl=3600, config=LlmLoadModelConfigDict(contextLength=4096, seed=42))
        max_tokens = maximum_tokens_answer(llm, answers, max_tokens)
        draft_model_key = None
        # if "qwen" in model:
        #     draft_model_key = "deepseek-r1-distill-qwen-1.5b"
        # '_to_history_content', 'content', 'load_config', 'model_info', 'parsed', 'prediction_config', 'stats', 'structured']
        raw = llm.respond(
            question,
            config=LlmPredictionConfigDict(
                maxTokens=max_tokens,
                temperature=temperature,
                repeatPenalty=repeatPenalty,  # This is particularly useful for preventing the model from repeating phrases or getting stuck in loops.
                # reasoningParsing=LlmReasoningParsingDict(
                #     enabled=False, # https://habr.com/ru/articles/1033808/
                #     startString=field(name="startString"),
                #     endString=field(name="endString")),
                draftModel=draft_model_key,
            ),
            response_format=OutputSchema,
        )
        parsed = _extract_first_json_object(raw) or {}

        return parsed, raw.stats.to_dict()

    if backend == "ollama_sdk":
        import ollama

        raw_resp = ollama.generate(
            model=model,
            prompt=prompt["messages"],
            options={"temperature": temperature, "num_predict": max_tokens},
            keep_alive="10m",
        )
        raw = (raw_resp or {}).get("response", "").strip()
        try:
            ollama.generate(model=model, prompt="", keep_alive=0)
        except Exception:
            pass
        parsed = _extract_first_json_object(raw) or {}
        return parsed, raw

    if backend == "lmstudio_rest":
        raw = _call_openai_compatible_completion(
            base_url="http://127.0.0.1:1234",
            model=model,
            prompt=prompt["messages"],
            temperature=temperature,
            maxTokens=max_tokens,
        )
        parsed = _extract_first_json_object(raw) or {}
        return parsed, raw

    if backend == "ollama_rest":
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt["messages"],
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        try:
            requests.post(url, json={"model": model, "prompt": "", "stream": False, "keep_alive": 0}, timeout=60)
        except Exception:
            pass
        parsed = _extract_first_json_object(raw) or {}
        return parsed, raw

    raise ValueError(f"Unknown backend: {backend}")


def _detect_backend() -> str:
    try:
        import lmstudio as _  # noqa: F401

        return "lmstudio_sdk"
    except Exception:
        pass

    try:
        import ollama as _  # noqa: F401

        return "ollama_sdk"
    except Exception:
        pass

    return "lmstudio_rest"


def _parse_model_entry(entry: str, default_backend: str) -> tuple[str, str]:
    if ":" in entry:
        prefix, name = entry.split(":", 1)
        prefix = prefix.strip().lower()
        if prefix in {"lmstudio_sdk", "ollama_sdk", "lmstudio_rest", "ollama_rest"}:
            return prefix, name.strip()
    return default_backend, entry


def _build_model_configs(settings: SetupSettings, test_mode: bool, max_tokens: int = 1024) -> list[ModelConfig]:
    """
    Собираем конфиги моделей, которые будут считать
    :param settings: настройки со списком моделей.
    :param test_mode:
    :return:
    """
    default_backend = os.getenv("LLM_BACKEND", "").strip().lower() or _detect_backend()

    llms = list(settings.llms)
    if test_mode:
        llms = llms[:2]

    configs: list[ModelConfig] = []
    for entry in llms:
        backend, model_name = _parse_model_entry(entry, default_backend)
        for thinking in (False,):  # (False, True):  todo только не думающая

            temperature = 0.0  # 0.2
            max_tokens = max_tokens
            if model_name == "google/gemma-4-26b-a4b":
                temperature = 0.3
            configs.append(
                ModelConfig(
                    backend=backend,
                    model=model_name,
                    thinking=thinking,
                    temperature=temperature,
                    maxTokens=max_tokens,
                )
            )

    return configs


def generate_answer(key, q, out_csv, encoding):
    try:
        parsed, raw = _generate_structured_answer(
            backend=key.backend,
            model=key.model,
            question=q["question"],
            context=q.get("context"),
            thinking=key.thinking,
            temperature=key.temperature,
            max_tokens=key.maxTokens,
            answers=q["correct_answer"],
        )
    except Exception as e:
        parsed = {}
        raw = ""
        logger.error(f"LLM error for {key.model}: {e}")

    answer_text = str(parsed.get("answer") or "").strip()
    confidence = parsed.get("confidence")
    reasoning = str(parsed.get("reasoning") or "").strip() if key.thinking else ""

    if isinstance(q["correct_answer"], list):
        is_correct = any(_evaluate_answer_complex(answer_text, ca) for ca in q["correct_answer"])
    else:
        is_correct = _evaluate_answer_complex(answer_text, str(q["correct_answer"]))

    row: dict[str, Any] = {
        "dataset": q["dataset"],
        "question_id": q["question_id"],
        "question": q["question"],
        "backend": key.backend,
        "model": str(key),
        "thinking": key.thinking,
        "answer": answer_text,
        "reasoning": reasoning,
        "confidence": confidence,
        "raw": raw,
        "context_used": q.get("context_used", False),
        "is_correct": is_correct,
    }

    _append_row_atomic(out_csv, row, encoding=encoding)


def _extract_first_json_object(text: str | PredictionResult) -> dict[str, Any] | Mapping[str, Any] | None:
    if isinstance(text, PredictionResult):
        if text.structured:
            return text.parsed
        else:
            return {"answer": text.content}

    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except Exception:
                    return None
                if isinstance(obj, dict):
                    return obj
                return None

    return None


def build_structured_params(question: str, context: str | None = None, thinking: bool = False) -> dict:
    """
    Возвращает параметры для structured output в LM Studio (OpenAI‑совместимый формат).
    Подходит для прямых HTTP‑запросов к /v1/chat/completions или для OpenAI‑клиентов.
    """

    # --- 1. Сообщения (промпт) ---
    # Здесь только полезная информация, без требований "return JSON".
    prompt_parts = []

    if context:
        prompt_parts.append(f"Context:\n{context}")

    prompt_parts.append(f"Question: {question}")

    # Если модель должна объяснять ход мыслей, можно мягко подсказать в тексте.
    if thinking:
        prompt_parts.append("Please provide a step-by-step reasoning in the 'reasoning' field.")

    user_message = "\n\n".join(prompt_parts)

    messages = [{"role": "user", "content": user_message}]

    # --- 2. JSON Schema (зависит от thinking) ---
    properties = {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    }
    required_fields = ["answer", "confidence"]

    if thinking:
        properties["reasoning"] = {"type": "string"}
        required_fields.append("reasoning")

    schema = {
        "type": "object",
        "properties": properties,
        "required": required_fields,
    }

    # --- 3. response_format для OpenAI‑совместимого API ---
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_response",  # любое уникальное имя
            "strict": True,  # строгий режим (рекомендуется)
            "schema": schema,
        },
    }

    # --- 4. Итоговый словарь параметров ---
    return {
        "messages": messages,
        "response_format": response_format,
    }


def _call_openai_compatible_completion(
    base_url: str, model: str, prompt: str, temperature: float, maxTokens: int
) -> str:
    url = base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "maxTokens": maxTokens,
    }
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    return (data.get("choices") or [{}])[0].get("text", "").strip()


def maximum_tokens_answer(model: lms.LLM, answers: str | list, max_tokens) -> int:
    token_count = 0
    if isinstance(answers, str):
        token_count = len(model.tokenize(answers))
    else:
        for answer in answers:
            token_count = max(token_count, len(model.tokenize(answer)))
    return max_tokens + token_count


def _atomic_write_csv(df: pd.DataFrame, path: Path, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    if path.exists():
        path.unlink()
    os.replace(tmp_path, path)


def _append_row_atomic(path: Path, row: dict[str, Any], encoding: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path)
        except Exception as e:
            logger.exception(e)
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    new_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    _atomic_write_csv(new_df, path, encoding=encoding)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = " ".join(text.split())
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


def _extract_numbers(text: str) -> list[str]:
    return [n for n in re.findall(r"\d+\.?\d*", text) if n]


def _evaluate_answer_complex(generated_answer: str, correct_answer: str) -> bool:
    gen_norm = _normalize_text(generated_answer)
    corr_norm = _normalize_text(correct_answer)

    if gen_norm == corr_norm:
        return True

    gen_tokens = set(gen_norm.split())
    corr_tokens = set(corr_norm.split())
    if corr_tokens and len(gen_tokens.intersection(corr_tokens)) / len(corr_tokens) >= 0.8:
        return True

    gen_nums = _extract_numbers(generated_answer)
    corr_nums = _extract_numbers(correct_answer)
    if gen_nums and corr_nums and set(gen_nums) == set(corr_nums):
        return True

    if _calculate_similarity(gen_norm, corr_norm) >= 0.85:
        return True

    if _is_substring_with_flexibility(gen_norm, corr_norm):
        return True

    return False


def _calculate_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    matches = 0
    max_len = max(len(s1), len(s2))
    s1_padded = s1.ljust(max_len)
    s2_padded = s2.ljust(max_len)
    for c1, c2 in zip(s1_padded, s2_padded, strict=False):
        if c1 == c2:
            matches += 1
    return matches / max_len


def _is_contained_with_tolerance(needle: str, haystack: str) -> bool:
    needle_words = set(needle.split())
    haystack_words = set(haystack.split())

    if len(needle_words) <= 2:
        return needle_words.issubset(haystack_words)

    intersection = needle_words.intersection(haystack_words)
    if len(needle_words) == 0:
        return True
    return len(intersection) / len(needle_words) >= 0.7


def _is_substring_with_flexibility(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    if len(needle) < 3:
        return needle == haystack
    if len(needle) <= len(haystack):
        return _is_contained_with_tolerance(needle, haystack)
    return _is_contained_with_tolerance(haystack, needle)
