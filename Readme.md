Весь код для ВКР.

main.py - основной файл, запускает все функции

mkdocs_rag_plugin - плагин








____
[generate_qna.py](src/custom_dataset/generate_qna.py) в этом файле создаются 200 вопросов, для проверки на реальном
датасете.
___

### Команды для запуска тестов из командной строки

- Базовый запуск со всеми плагинами (использует настройки из pytest.ini)
  ```shell
  pytest
  ```

- Запуск с подробным выводом и отключением параллелизма для отладки
  ```shell
  pytest -v -s -n 0
  ```

- Запуск конкретного теста
  ```shell
  pytest tests/test_extract_and_download.py::TestExtractAndDownload::test_successful_download
  ```
- Запуск с фильтром по имени
  ```shell
  pytest -k "test_env"
  ```
- Запуск без покрытия (быстрее работает)
  ```shell
  pytest --no-cov
  ```

- Запуск только с пересчетом покрытия
  ```shell
  pytest --cov --cov-report=term
  ```
- Запуск в параллельном режиме с явным указанием количества процессов
  ```shell
  pytest -n 4
  ```
- Запуск с рандомизацией порядка тестов (для выявления зависимостей)
  ```shell
  pytest --randomly-seed=12345
  ```
- Запуск с генерацией отчета в HTML и открытием в браузере
  ```shell
  pytest && python -m http.server --directory htmlcov 8080
  ```
- Запуск в режиме CI (без интерактива, с цветовым выводом)
  ```shell
  pytest --color=yes --tb=short
  ```
- Запуск с профайлингом (требует pytest-profiling)
  ```shell
  pytest --profile
  ```
- Комбинированный запуск для CI/CD
  ```shell
  pytest \
    --cov=extract_and_download \
    --cov-report=xml \
    --cov-report=term-missing \
    --randomly-seed=42 \
    -n auto \
    -q
  ```
- Проверка, что переменная test=true установлена во время тестов
  ```shell
  pytest -v -s -k "test_env" --tb=short
  ```
  ```shell
  pre-commit run --all-files
  ```

### UV commands

```shell
uv audit --preview-features audit
```

```shell
uv sync --all-extras
```

```shell
uv sync --no-dev
```

```shell
uv lock --upgrade
```
