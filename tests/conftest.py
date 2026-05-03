# tests/conftest.py
"""
Глобальные фикстуры и конфигурация pytest.
Автоматически подгружается pytest при запуске тестов.
"""

import os
import sys
from pathlib import Path

from pytest import fixture

pytest_plugins = ["tests.fixtures.extract_fixtures"]


# # Фикстура для установки переменной окружения test=true
# @pytest.fixture(autouse=True)
# def set_test_env(monkeypatch):
#     """Автоматически устанавливает test=true в окружение для всех тестов"""
#     monkeypatch.setenv("test", "true")
#     yield
#     # Очистка после теста (опционально)
#     monkeypatch.delenv("test", raising=False)


def pytest_configure(config):
    """Конфигурация перед запуском тестов"""
    # Добавляем корень проекта в sys.path для импортов
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Устанавливаем переменную окружения на уровне сессии
    os.environ["test"] = "true"


def pytest_unconfigure(config):
    """Очистка после завершения тестовой сессии"""
    # Опционально: удалить переменную после всех тестов
    os.environ.pop("test", None)
    pass


@fixture(scope="session")
def project_root():
    """Фикстура с путем к корню проекта"""
    return Path(__file__).parent.parent


# # conftest.py для интеграции с pytest-async (если в скрипте появятся async-функции)
#
# @pytest.fixture
# def async_mock(mocker):
#     """Фикстура для мокирования async-функций"""
#     def _async_mock(return_value=None, side_effect=None):
#         mock = mocker.AsyncMock()
#         if return_value is not None:
#             mock.return_value = return_value
#         if side_effect is not None:
#             mock.side_effect = side_effect
#         return mock
#     return _async_mock
