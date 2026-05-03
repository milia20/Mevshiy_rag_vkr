import logging
import os
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

import loguru
from loguru import logger

# Контекстные переменные для отслеживания запроса
request_id_context: ContextVar[str | None] = ContextVar("request_id", default="fast_api")
pid_context: ContextVar[int | None] = ContextVar("pid", default=None)


class InterceptHandler(logging.Handler):  # pylint: disable=too-few-public-methods
    """
    Интеграция loguru с uvicorn.
    Default handler from examples in loguru documentation.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    https://pawamoy.github.io/posts/unify-logging-for-a-gunicorn-uvicorn-app/
    """

    def emit(self, record: logging.LogRecord):
        # Get corresponding loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    def write(self, message):
        """
        Логируем ошибки самого Python
        """
        message = message.strip().strip("^~")  # ^ постоянно встречается как одиночный символ в строке
        # Преобразуем пути в кликабельные ссылки для PyCharm

        try:
            if message:
                logger.warning(message)
        except Exception as e:
            print("Exception loguru:", message, e)
            # Очистка файла из-за переполнения loguru с ограничением по размеру. (Повторить не получилось win)
            with open("app.log", "w"):
                pass
            logger.critical(str(e) + " произошла очистка логов")
            logger.warning(message)


def is_jupyter_notebook() -> bool:
    """
    Проверяем что есть возможность не только получить IPython, но и проверка, что это не интерактивный python терминал.
    :return: Внутри jupyter True False
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        # Проверяем, что это именно jupyter, а не ipython терминал PyCharm (PyDevTerminalInteractiveShell)
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def request_id_filter(record: "loguru.Record") -> None:
    """
    Функция patcher, вставляет в запись loger значения из этой среды
    :param record: TypedDict заполняющий logger
    """
    record["extra"]["request_id"] = request_id_context.get()
    record["extra"]["pid"] = pid_context.get()


def generate_short_request_id() -> str:
    """Генерация короткого уникального ID для запроса."""
    return str(uuid.uuid4().hex)[:8]


def bind_context(request_id: str | None = None, pid: int | None = None):
    """Привязка контекста к логгеру."""
    if request_id is None:
        request_id = generate_short_request_id()

    if pid is None and pid_context.get():
        pid = os.getpid()

    request_id_context.set(request_id)
    pid_context.set(pid)
    return request_id


def unbind_context():
    """Отвязка контекста."""
    request_id_context.set(None)
    pid_context.set(None)


def get_request_id() -> str | None:
    """
    Получить актуальное значение pid процесса в логере
    :return: PID
    """
    return request_id_context.get()


log_level = logging.INFO
logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Перестраиваем стандартные логгеры. Удаляем все существующие хендлеры
for _name in logging.root.manager.loggerDict.keys():
    logging.getLogger(_name).handlers = []
    logging.getLogger(_name).propagate = True

# Добавляем наш перехватчик в корневой логгер
if not is_jupyter_notebook():
    logging.basicConfig(handlers=[InterceptHandler()], level=log_level)

sys.stderr = InterceptHandler()  # Логируем ошибки других библиотек Python, например sqlalchemy, ее класс DBAPI

logger.configure(extra={"request_id": "fast_api", "pid": os.getpid()}, patcher=request_id_filter)
logger.remove()

logger.add(
    sink=Path(__file__).parent / "logs" / "app.log",
    rotation="00:00",
    retention="10 days",
    compression="zip",
    level=log_level,
    colorize=False,
    backtrace=True,
    diagnose=True,
    serialize=True,
    enqueue=True,
    format=logger_format,
)

# Меняем уровень логирования в консоли.
logger.add(sink=sys.stdout, level="TRACE", format=logger_format)
