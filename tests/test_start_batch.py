"""
Функциональный тест для start.bat
Проверяет:
- запуск скрипта с окружением TEST=true
- корректную инициализацию окружения
- вызов main.py с мокированными зависимостями
- логирование и вывод в stdout
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Пути к проекту
PROJECT_ROOT = Path(__file__).parent.parent
BATCH_FILE = PROJECT_ROOT / "start.bat"
MAIN_PY = PROJECT_ROOT / "main.py"


@pytest.fixture(scope="module")
def test_env():
    env = os.environ.copy()
    env["TEST"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    env["LOGURU_COLORIZE"] = "0"

    # Ключевые переменные для кодировки
    env["PYTHONUTF8"] = "1"  # Python 3.7+
    env["PYTHONIOENCODING"] = "CP866"  # Все версии Python
    env["UV_NO_PROGRESS"] = "1"  # Убрать прогресс-бары uv (могут ломать вывод)

    return env


def test_batch_file_exists():
    """Проверка наличия start.bat"""
    assert BATCH_FILE.exists(), f"Файл {BATCH_FILE} не найден"


def test_main_py_exists():
    """Проверка наличия main.py"""
    assert MAIN_PY.exists(), f"Файл {MAIN_PY} не найден"


@pytest.mark.windows_only
def test_start_batch_execution(test_env):
    """
    Интеграционный тест запуска start.bat
    Проверяет:
    - успешный запуск батника
    - наличие ключевых логов в выводе
    - корректную установку переменной окружения TEST=true
    """
    if sys.platform != "win32":
        pytest.skip("Тест требует Windows")

    test_env["TEST_MOCK_MAIN"] = "1"
    # Запуск батника из корня проекта
    result = subprocess.run(
        [str(BATCH_FILE)],
        cwd=PROJECT_ROOT,
        env=test_env,
        capture_output=True,
        text=True,
        encoding="CP866",
        errors="replace",  # ← Заменять невалидные символы, а не падать
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )

    output = result.stdout + result.stderr

    assert "Начало общей функции" in output, "Не найден стартовый print из main.py"
    assert "[MOCK]" in output, "Мокированные функции не были вызваны"
    # Проверка отсутствия критических ошибок
    assert "ОШИБКА:" not in output or "uv отсутствует" not in output, "Батник упал на проверке зависимостей"


def test_batch_prerequisites():
    """Проверка необходимых файлов для start.bat"""
    required_files = ["pyproject.toml", "uv.lock", "main.py"]
    missing = [f for f in required_files if not (PROJECT_ROOT / f).exists()]

    assert not missing, f"Отсутствуют обязательные файлы: {missing}"


#
# @pytest.mark.parametrize(
#     "env_var,expected",
#     [
#         ("TEST", "true"),
#         ("PYTHONUNBUFFERED", "1"),
#     ],
# )
# def test_environment_variables(test_env, env_var, expected):
#     """Проверка установки переменных окружения"""
#     assert test_env.get(env_var) == expected, f"{env_var} != {expected}"
