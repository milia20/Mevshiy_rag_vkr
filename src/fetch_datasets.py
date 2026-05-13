import os
import re
import subprocess
import sys
import urllib.request
from http import HTTPStatus
from pathlib import Path

import requests
from loguru import logger

from src.config import SetupSettings


def parse_github_url(url: str) -> dict | None:
    """
    Парсит GitHub URL с тегом/веткой и путем к папке.

    Поддерживаемые форматы:
    - https://github.com/org/repo/tree/tag/path/to/folder
    - https://github.com/org/repo/tree/branch/path/to/folder

    Args:
        url: GitHub URL с тегом/веткой и путем

    Returns:
        dict с ключами: owner, repo, ref (tag/branch), path или None
    """
    # Паттерн для GitHub URL с tree/branch/path
    pattern = r"https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)"
    match = re.match(pattern, url)

    if match:
        return {"owner": match.group(1), "repo": match.group(2), "ref": match.group(3), "path": match.group(4)}
    return None


def sparse_checkout_clone(github_url: str, target_dir: Path, force: bool = False) -> bool:
    """
    Клонирует репозиторий используя современный метод sparse checkout.

    Args:
        github_url: GitHub URL с тегом/веткой и путем (например: https://github.com/org/repo/tree/tag/path)
        target_dir: Целевая директория для сохранения
        force: Принудительное перекачивание

    Returns:
        bool: True если успешно, False если ошибка
    """
    parsed = parse_github_url(github_url)
    if not parsed:
        logger.error(f"Не удалось распарсить GitHub URL: {github_url}")
        return False

    owner, repo, ref, path = parsed["owner"], parsed["repo"], parsed["ref"], parsed["path"]
    clone_url = f"https://github.com/{owner}/{repo}.git"
    repo_name = f"{owner}_{repo}"
    temp_dir = target_dir.parent / f"temp_{repo_name}"

    # Если целевая папка уже существует и не force, пропускаем
    if target_dir.exists() and not force:
        print(f"✅ Папка уже существует: {target_dir} (пропущено)")
        return True

    # Удаляем существующие папки если force
    if target_dir.exists():
        import shutil

        shutil.rmtree(target_dir)
    if temp_dir.exists():
        import shutil

        shutil.rmtree(temp_dir)

    try:
        print(f"Клонирование {clone_url} с sparse checkout для пути: {path} (ветка: {ref})")

        # 1. Клонируем репозиторий без автоматической выгрузки файлов и без истории
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", clone_url, str(temp_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        # 2. Включаем режим разреженного вывода (sparse checkout)
        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"], check=True, cwd=temp_dir, capture_output=True, text=True
        )

        # 3. Указываем нужную папку
        subprocess.run(
            ["git", "sparse-checkout", "set", path], check=True, cwd=temp_dir, capture_output=True, text=True
        )

        # 4. Выгружаем файлы для указанной папки
        subprocess.run(["git", "checkout", ref], check=True, cwd=temp_dir, capture_output=True, text=True)

        # Перемещаем папку в целевую директорию
        source_path = temp_dir / path
        if source_path.exists():
            import shutil

            shutil.move(str(source_path), str(target_dir))
            print(f"✅ Успешно скачано: {target_dir}")
        else:
            logger.error(f"Путь {path} не найден в репозитории")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка выполнения git команды: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Ошибка при sparse checkout: {e}")
        return False
    finally:
        # Удаляем временную папку
        if temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir)

    return True


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

    # Создаем папку docs в корне проекта
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    all_urls = list(urls)  # Копируем список для безопасной модификации

    for url in all_urls:
        # Проверяем, является ли URL GitHub репозиторием с тегом/веткой и путем
        if parse_github_url(url):
            # Используем sparse checkout для GitHub репозиториев
            parsed = parse_github_url(url)
            subdir_name = f"{parsed['owner']}_{parsed['repo']}"
            target_dir = docs_dir / subdir_name / parsed["path"].split("/")[-1]

            if sparse_checkout_clone(url, target_dir, force):
                continue
            else:
                logger.error(f"Не удалось скачать {url} через sparse checkout")
                sys.exit(1)

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
        elif "CoSQA" in url:
            subdir = datasets_dir / "CoSQA"
        else:
            subdir = datasets_dir

        subdir.mkdir(exist_ok=True)
        file_path = subdir / filename

        if file_path.exists() and not force:
            print(f"✅ Файл уже существует: {file_path} (пропущено)")
            continue
        else:
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
                    logger.trace(f"Файл: {item['name']}, URL для скачивания: {item['download_url']}")
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
