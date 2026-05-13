"""
Асинхронный клиент для взаимодействия с LLM.

Модуль предоставляет класс для генерации ответов с использованием
OpenAI-совместимых API (LM Studio, Ollama) с поддержкой стриминга.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import httpx
from src.core.config import settings
from src.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Асинхронный клиент для LLM провайдеров.

    Поддерживает OpenAI-совместимые API с fallback между LM Studio и Ollama.
    Реализует exponential backoff для повторных попыток.

    Атрибуты:
        base_url: Базовый URL API.
        model_name: Название модели.
        timeout: Таймаут запроса в секундах.
        max_retries: Максимальное количество повторных попыток.
    """

    # Приоритетные endpoint'ы для fallback
    FALLBACK_ENDPOINTS = [
        {"base_url": "http://localhost:1234/v1", "provider": "lmstudio"},
        {"base_url": "http://localhost:11434/v1", "provider": "ollama"},
    ]

    RUSSIAN_SYSTEM_PROMPT = """Ты — помощник по технической документации. Отвечай ТОЛЬКО на основе предоставленного контекста.
Если в контексте нет информации для ответа, напиши: "Не удалось найти ответ в документации."

Контекст:
{formatted_contexts}

Вопрос: {query}
Ответ:"""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Инициализация LLM клиента.

        Args:
            base_url: Базовый URL API. Если не указан, используется из настроек.
            model_name: Название модели. Если не указано, используется из настроек.
            timeout: Таймаут запроса. Если не указан, используется из настроек.
            max_retries: Максимум повторных попыток. Если не указан, используется из настроек.
        """
        self.base_url = base_url or settings.llm_base_url
        self.model_name = model_name or settings.llm_model
        self.timeout = timeout or settings.request_timeout
        self.max_retries = max_retries or settings.max_retries

        logger.info(
            f"Инициализация LLMClient: model={self.model_name}, "
            f"base_url={self.base_url or 'auto-detect'}, timeout={self.timeout}s"
        )

        self._client: httpx.AsyncClient | None = None
        self._current_endpoint: dict[str, str] | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Получить или создать HTTP клиент.

        Returns:
            Экземпляр httpx.AsyncClient.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def _detect_endpoint(self) -> dict[str, str]:
        """
        Автоматически определить доступный endpoint.

        Проверяет fallback endpoint'ы в порядке приоритета.

        Returns:
            Словарь с информацией об endpoint.

        Raises:
            LLMError: Если ни один endpoint не доступен.
        """
        client = await self._get_client()

        for endpoint in self.FALLBACK_ENDPOINTS:
            try:
                response = await client.get(f"{endpoint['base_url']}/models", timeout=5.0)
                if response.status_code == 200:
                    logger.info(f"Обнаружен доступный endpoint: {endpoint['provider']}")
                    self._current_endpoint = endpoint
                    return endpoint
            except Exception:
                logger.debug(f"Endpoint {endpoint['base_url']} недоступен")
                continue

        msg = "Ни один LLM endpoint не доступен (LM Studio / Ollama)"
        logger.error(msg)
        raise LLMError(msg)

    async def _make_request_with_retry(
        self,
        payload: dict,
        stream: bool = False,
    ) -> httpx.Response | httpx.AsyncStream:
        """
        Выполнить запрос с exponential backoff.

        Args:
            payload: Тело запроса.
            stream: Включить ли стриминг.

        Returns:
            Response или AsyncStream от httpx.

        Raises:
            LLMError: Если все попытки исчерпаны.
        """
        client = await self._get_client()

        # Определяем базовый URL
        base_url = self.base_url
        if not base_url:
            endpoint = self._current_endpoint or await self._detect_endpoint()
            base_url = endpoint["base_url"]

        url = f"{base_url}/chat/completions"

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                if stream:
                    response = await client.stream(
                        "POST",
                        url,
                        json=payload,
                    )
                    return response
                else:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return response

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"HTTP ошибка (попытка {attempt + 1}/{self.max_retries}): {e}")
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"Ошибка запроса (попытка {attempt + 1}/{self.max_retries}): {e}")

                # Попытка переключиться на fallback endpoint
                if not self.base_url and attempt < self.max_retries - 1:
                    try:
                        new_endpoint = await self._detect_endpoint()
                        base_url = new_endpoint["base_url"]
                        url = f"{new_endpoint['base_url']}/chat/completions"
                        logger.info(
                            f"Переключение на fallback endpoint: {new_endpoint['provider']}"
                        )
                    except LLMError:
                        pass

            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = (2**attempt) * 0.5  # 0.5s, 1s, 2s, ...
                logger.info(f"Ожидание {wait_time}s перед повторной попыткой...")
                await asyncio.sleep(wait_time)

        msg = f"Все {self.max_retries} попыток исчерпаны"
        logger.error(msg)
        raise LLMError(msg, str(last_error) if last_error else None)

    async def generate(
        self,
        query: str,
        contexts: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """
        Сгенерировать ответ на вопрос.

        Args:
            query: Вопрос пользователя.
            contexts: Список контекстов (словари с content/metadata).
            temperature: Температура генерации.
            max_tokens: Максимальное количество токенов в ответе.

        Returns:
            Сгенерированный ответ.

        Raises:
            LLMError: Если произошла ошибка генерации.
        """
        # Форматируем контекст
        formatted_contexts = "\n\n".join(
            [f"[Источник: {ctx.get('url', 'N/A')}]\n{ctx.get('content', '')}" for ctx in contexts]
        )

        # Создаем промпт на русском
        user_prompt = self.RUSSIAN_SYSTEM_PROMPT.format(
            formatted_contexts=formatted_contexts,
            query=query,
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты — полезный ассистент по технической документации.",
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        logger.info(f"Генерация ответа для запроса: {query[:50]}...")

        try:
            response = await self._make_request_with_retry(payload, stream=False)
            data = response.json()

            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not answer:
                msg = "Пустой ответ от LLM"
                logger.error(msg)
                raise LLMError(msg)

            logger.info(f"Сгенерирован ответ длиной {len(answer)} символов")
            return answer

        except LLMError:
            raise
        except Exception as e:
            msg = f"Ошибка обработки ответа LLM: {e}"
            logger.error(msg)
            raise LLMError(msg, str(e)) from e

    async def generate_stream(
        self,
        query: str,
        contexts: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Сгенерировать ответ с потоковой передачей.

        Args:
            query: Вопрос пользователя.
            contexts: Список контекстов.
            temperature: Температура генерации.
            max_tokens: Максимальное количество токенов.

        Yields:
            Части ответа (tokens/chunks).

        Raises:
            LLMError: Если произошла ошибка генерации.
        """
        formatted_contexts = "\n\n".join(
            [f"[Источник: {ctx.get('url', 'N/A')}]\n{ctx.get('content', '')}" for ctx in contexts]
        )

        user_prompt = self.RUSSIAN_SYSTEM_PROMPT.format(
            formatted_contexts=formatted_contexts,
            query=query,
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты — полезный ассистент по технической документации.",
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        logger.info(f"Потоковая генерация для запроса: {query[:50]}...")

        try:
            async with await self._make_request_with_retry(payload, stream=True) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Убираем "data: "
                        if data_str.strip() == "[DONE]":
                            break

                        import json

                        try:
                            data = json.loads(data_str)
                            chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue

        except LLMError:
            raise
        except Exception as e:
            msg = f"Ошибка потоковой генерации: {e}"
            logger.error(msg)
            raise LLMError(msg, str(e)) from e

    async def close(self) -> None:
        """Закрыть HTTP клиент."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("LLM клиент закрыт")


# Singleton instance
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """
    Получить экземпляр LLM клиента (singleton).

    Returns:
        Экземпляр LLMClient.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
