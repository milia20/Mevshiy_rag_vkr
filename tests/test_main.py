import pytest


@pytest.fixture(scope="function")
def test_env(monkeypatch):
    """Устанавливает переменные окружения для теста"""
    monkeypatch.setenv("TEST", "true")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setenv("LOGURU_COLORIZE", "0")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "CP866")
    monkeypatch.setenv("UV_NO_PROGRESS", "1")
    # Убеждаемся, что TEST_MOCK_MAIN не установлена (чтобы вызывалась настоящая main)
    monkeypatch.delenv("TEST_MOCK_MAIN", raising=False)


# def test_main_handles_exception_gracefully(test_env):
#     """
#     Альтернативный вариант: проверка, что main не падает с исключениями.
#     (Если внутри main ничего не выбрасывает, ничего дополнительно проверять не нужно)
#     """
#     try:
#         main()
#     except Exception as e:
#         pytest.fail(f"main() выбросила исключение: {e}")
#     # Если дошли сюда – успех
#     assert True
