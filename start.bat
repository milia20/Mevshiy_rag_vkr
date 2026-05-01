@echo off
chcp 65001 >nul
setlocal

:: Проверка uv
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: uv отсутствует в PATH.
    goto :wait_exit
)

:: Проверка конфигурационных файлов
if not exist "pyproject.toml" (
    echo ОШИБКА: pyproject.toml не найден.
    goto :wait_exit
)
if not exist "uv.lock" (
    echo ОШИБКА: uv.lock не найден.
    goto :wait_exit
)

:: Проверка поддержки tomllib (требует Python >= 3.11)
python -c "import tomllib" >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Требуется Python 3.11+ или установленный модуль tomli.
    echo Обновите Python или добавьте его в PATH.
    goto :wait_exit
)

:: Извлечение URL
echo Извлечение ссылок из pyproject.toml...
python -c "import tomllib; f=open('pyproject.toml','rb'); d=tomllib.load(f); [print(u) for u in d.get('tool',{}).get('remote_scripts',{}).get('urls',[])]" > _tmp_urls.txt
if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось прочитать [tool.remote_scripts.urls].
    goto :wait_exit
)

:: Загрузка файлов
for /f "delims=" %%i in (_tmp_urls.txt) do (
    echo Загрузка: %%i
    curl -fSL -O "%%i"
    if %errorlevel% neq 0 (
        echo ОШИБКА ЗАГРУЗКИ: %%i
        del _tmp_urls.txt
        goto :wait_exit
    )
)
del _tmp_urls.txt

:: Настройка окружения
echo Создание виртуальной среды...
uv venv .venv
if %errorlevel% neq 0 goto :wait_exit

echo Установка зависимостей...
uv sync --frozen
if %errorlevel% neq 0 goto :wait_exit

echo Запуск скриптов...
.venv\Scripts\python.exe main_calc.py

echo ВЕСЬ КОД ВЫПОЛНЕН.
:wait_exit
set /p "CHOICE=Введите Y и нажмите Enter для закрытия окна: "
if /i "%CHOICE%"=="Y" exit /b 0
goto :wait_exit