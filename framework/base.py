import time
import threading
import pathlib
import enum


class LogLevel(enum.IntEnum):
    ERROR = 1
    WARNING = 2
    INFO = 3
    VERBOSE = 4


__level = LogLevel.INFO


def set_log_level(level: LogLevel):
    global __level
    __level = level


def get_log_level() -> LogLevel:
    return __level


def log_verbose() -> bool:
    return __level >= LogLevel.VERBOSE



def log(fmt, *args):
    print(time.strftime("[%H:%M:%S]", time.localtime()) + "[" + threading.current_thread().name + "] " + str(fmt), *args)


def get_files(path, pattern):
    files = []
    for path in pathlib.Path(path).rglob(pattern):
        files.append(path.as_posix())
        log(path.as_posix())
    return files
