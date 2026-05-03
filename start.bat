@echo off
@REM chcp 65001 >nul
setlocal

:: Проверка uv
uv --version >nul 2>&1
if %errorlevel% neq 0 ( echo ОШИБКА: uv отсутствует в PATH. & goto :wait_exit )
if not exist "pyproject.toml" ( echo ОШИБКА: pyproject.toml не найден. & goto :wait_exit )
if not exist "uv.lock" ( echo ОШИБКА: uv.lock не найден. & goto :wait_exit )
if not exist "main.py" ( echo ОШИБКА: main.py не найден. & goto :wait_exit )

python -c "import tomllib" >nul 2>&1
if %errorlevel% neq 0 ( echo ОШИБКА: Требуется Python 3.11+. & goto :wait_exit )

:: Настройка окружения (повторное использование существующего)
if exist ".venv" (
    echo Виртуальная среда .venv уже существует. Пропуск создания.
) else (
    echo Создание виртуальной среды...
    uv venv .venv
    if %errorlevel% neq 0 goto :wait_exit
)

echo Установка пакетов...
uv sync --frozen --link-mode=copy --all-extras
if %errorlevel% neq 0 goto :wait_exit


echo Запуск скриптов...
.venv\Scripts\python.exe main.py
@REM if %errorlevel% neq 0 goto :wait_exit
@REM .venv\Scripts\python.exe main_calc.py

echo ВЕСЬ КОД ВЫПОЛНЕН.
:wait_exit
@REM set /p "CHOICE=Введите Y и нажмите Enter для закрытия окна: "
@REM if /i "%CHOICE%"=="Y" exit /b 0
@REM goto :wait_exit
