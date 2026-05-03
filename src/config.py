# config.py
from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class SetupSettings(BaseModel):
    """
    Модель для секции [tool.setup_settings] из pyproject.toml
    """

    remote_files_urls: list[str] = Field(default_factory=list, description="Список URL для загрузки удалённых файлов")
    llms: list[str] = Field(default_factory=list, description="Список идентификаторов LLM-моделей")

    test: bool = True  # os.getenv("TEST", False)

    @field_validator("remote_files_urls")
    @classmethod
    def validate_urls(cls, urls: list[str]) -> list[str]:
        """Валидация: каждый элемент должен быть валидным HTTP/HTTPS URL"""
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme not in ("https", "http"):
                raise ValueError(f"Invalid URL scheme in '{url}': must be http or https")
            if not parsed.netloc:
                raise ValueError(f"Invalid URL format: '{url}'")
        return urls

    @field_validator("llms")
    @classmethod
    def validate_llm_format(cls, models: list[str]) -> list[str]:
        """Валидация: формат 'организация/модель' или допустимый идентификатор"""
        for model in models:
            if "/" not in model or len(model.split("/")) != 2:
                raise ValueError(f"LLM identifier should be in 'provider/model', got: '{model}'")
        return models


class PyProjectConfig(BaseModel):
    """
    Корневая модель для парсинга pyproject.toml
    """

    model_config = ConfigDict(populate_by_name=True)  # для совместимости с tomllib

    tool: ToolSection


class ToolSection(BaseModel):
    setup_settings: SetupSettings | None = None


def load_setup_settings(pyproject_path: Path | str | None = None) -> SetupSettings:
    """
    Загружает и валидирует настройки из pyproject.toml

    Args:
        pyproject_path: Путь к pyproject.toml. Если None, ищет в текущей директории и выше.

    Returns:
        SetupSettings: Валидированные настройки

    Raises:
        FileNotFoundError: Если файл не найден
        ValidationError: Если данные не проходят валидацию
        tomllib.TOMLDecodeError: Если файл содержит невалидный TOML
    """
    if pyproject_path is None:
        # Поиск pyproject.toml вверх по дереву директорий
        current = Path.cwd()
        for parent in [current, *current.parents]:
            candidate = parent / "pyproject.toml"
            if candidate.exists():
                pyproject_path = candidate
                break
        else:
            raise FileNotFoundError("pyproject.toml not found in current or parent directories")
    else:
        pyproject_path = Path(pyproject_path)
        if not pyproject_path.exists():
            raise FileNotFoundError(f"File not found: {pyproject_path}")

    with cast(BinaryIO, pyproject_path.open("rb")) as f:
        data = tomllib.load(f)

    # Обёртка в корневую модель для корректного парсинга вложенной структуры
    root = PyProjectConfig.model_validate({"tool": data.get("tool", {})})

    if root.tool.setup_settings is None:
        # Возвращаем пустые настройки с дефолтными значениями
        # Или можно выбросить ошибку: raise ValueError("[tool.setup_settings] not found in pyproject.toml")
        return SetupSettings()

    return root.tool.setup_settings


# Пример использования
if __name__ == "__main__":
    try:
        settings = load_setup_settings()
        print(f"✓ Загружено {len(settings.remote_files_urls)} URL")
        print(f"✓ Загружено {len(settings.llms)} LLM-моделей")
        print("\nURLs:")
        for url in settings.remote_files_urls:
            print(f"  - {url}")
        print("\nLLMs:")
        for llm in settings.llms:
            print(f"  - {llm}")
    except ValidationError as e:
        print(f"✗ Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
