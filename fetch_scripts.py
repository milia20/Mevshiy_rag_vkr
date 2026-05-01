import os
import sys
import tomllib
import urllib.request


def extract_and_download(toml_path: str = "pyproject.toml") -> None:
    """
    Встроенный парсер TOML отсутствует в CMD/PowerShell. Извлечение требует Python. Используйте tomllib (Python 3.11+) или tomli (для более старых версий).
    :param toml_path: Путь к файлу с настройками
    """
    try:
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

    for url in urls:
        filename = os.path.basename(url)
        if not filename.endswith(".py"):
            filename += ".py"
        print(f"Загрузка: {url} -> {filename}")
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"ОШИБКА СЕТИ: {url} -> {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    extract_and_download()
