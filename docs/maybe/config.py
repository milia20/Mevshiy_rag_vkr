# config.py
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, HttpUrl


class SetupSettings(BaseModel):
    """
    Модель для секции [tool.setup_settings] из pyproject.toml
    """

    remote_files_urls: list[HttpUrl] = Field(
        default_factory=list, description="Список URL для загрузки удалённых файлов"
    )
    llms: list[str] = Field(default_factory=list, description="Список идентификаторов LLM-моделей")

    # @field_validator("remote_files_urls")
    # @classmethod
    # def validate_urls(cls, urls: list[str]) -> list[str]:
    #     """Валидация: каждый элемент должен быть валидным HTTP/HTTPS URL"""
    #     for url in urls:
    #         parsed = urlparse(url)
    #         if parsed.scheme not in ("https", "http"):
    #             raise ValueError(f"Invalid URL scheme in '{url}': must be http or https")
    #         if not parsed.netloc:
    #             raise ValueError(f"Invalid URL format: '{url}'")
    #     return urls

    @field_validator("llms")
    @classmethod
    def validate_llm_format(cls, models: list[str]) -> list[str]:
        """Валидация: формат 'организация/модель' или допустимый идентификатор"""
        for model in models:
            if "/" not in model or len(model.split("/")) != 2:
                raise ValueError(f"LLM identifier should be in 'provider/model', got: '{model}'")
        return models

    @classmethod
    def from_pyproject(cls, path: str = "pyproject.toml") -> "SetupSettings":
        """Читает pyproject.toml и извлекает [tool.setup_settings]."""
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)

        tool_settings = data.get("tool", {}).get("setup_settings", {})
        if not tool_settings:
            raise ValueError("Секция [tool.setup_settings] не найдена в pyproject.toml")

        return cls(**tool_settings)


# Загрузка конфигурации из pyproject.toml
settings = SetupSettings.from_pyproject("../pyproject.toml")

print(settings.remote_files_urls)
print(settings.llms)
