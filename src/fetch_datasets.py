import os
import sys
import urllib.request
from http import HTTPStatus
from pathlib import Path

import requests
from loguru import logger

from src.config import SetupSettings


def extract_and_download(
    settings: SetupSettings | None = None, toml_path: str | Path = "pyproject.toml", force: bool = False
) -> None:
    """
    Загружает файлы из настроек SetupSettings или из TOML файла.

    Args:
        settings: Экземпляр SetupSettings с URL для загрузки. Если None, загружает из TOML
        toml_path: Путь к файлу с настройками (используется только если settings=None)
        force: Принудительное перекачивание
    """
    if settings is None:
        # Обратная совместимость: загружаем из TOML
        try:
            import tomllib

            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            print(f"ОШИБКА ЧТЕНИЯ TOML: {e}", file=sys.stderr)
            sys.exit(1)
        path = ["tool", "setup_settings", "remote_files_urls"]
        urls = data.get(path[0], {}).get(path[1], {}).get(path[2], [])
        if not urls:
            print(f"ОШИБКА: Секция {path[0]}.{path[1]}.{path[2]}] пуста или отсутствует.", file=sys.stderr)
            sys.exit(1)
    else:
        urls = settings.remote_files_urls
        if not urls:
            print("ОШИБКА: Список URL в настройках пуст.", file=sys.stderr)
            sys.exit(1)

    # Создаем папку datasets в корне проекта
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    all_urls = list(urls)  # Копируем список для безопасной модификации

    for url in all_urls:
        filename = os.path.basename(url)

        # Обработка GitHub API URL
        if "api.github.com" in url:
            github_files = github_get_first_1000_files(url)
            if github_files:
                all_urls.extend([file["download_url"] for file in github_files])
            continue

        if not filename:
            filename = url.split("/")[-1]

        # Определяем подпапку на основе URL
        if "ru_rag_test_dataset" in url:
            subdir = datasets_dir / "ru_rag_test_dataset"
        elif "RuBQ" in url:
            subdir = datasets_dir / "RuBQ"
        elif "natural_questions" in url:
            subdir = datasets_dir / "natural_questions"
        else:
            subdir = datasets_dir

        subdir.mkdir(exist_ok=True)
        file_path = subdir / filename

        if file_path.exists() and not force:
            print(f"✅ Файл уже существует: {file_path} (пропущено)")
            continue

        print(f"Загрузка: {url} -> {file_path}")
        try:
            urllib.request.urlretrieve(url, file_path)
        except Exception as e:
            logger.warning(f"ОШИБКА СЕТИ: {url} -> {e}", file=sys.stderr)
            sys.exit(1)


def github_get_first_1000_files(git_repo_url: str) -> list | None:
    """
    API имеет ограничение в 1000 файлов на директорию.
    Если файлов больше, понадобится [Git Trees API](https://docs.github.com/en/rest/repos/contents?apiVersion=2026-03-10#get-repository-content).

    Args:
        git_repo_url: URL GitHub API для содержимого репозитория

    Returns:
        list: Список словарей с информацией о файлах или None при ошибке
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    try:
        response = requests.get(git_repo_url, headers=headers)
        find_files = []
        max_files_response = 1000

        if response.status_code == HTTPStatus.OK:
            contents = response.json()
            if len(contents) == max_files_response:
                logger.warning("Превышено максимальное количество файлов на директорию. Используй Git Trees API")
            for item in contents:
                if item["type"] == "file":
                    find_files.append(item)
                    print(f"Файл: {item['name']}, URL для скачивания: {item['download_url']}")
            return find_files
        else:
            logger.warning(f"Ошибка получения списка файлов: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Ошибка при запросе к GitHub API: {e}")
        return None


if __name__ == "__main__":
    from src.config import load_setup_settings

    settings = load_setup_settings()
    extract_and_download(settings)
