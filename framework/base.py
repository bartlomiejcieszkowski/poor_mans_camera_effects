import time
import threading
import pathlib


def log(fmt, *args):
    print(time.strftime("[%H:%M:%S]", time.localtime()) + "[" + threading.current_thread().name + "] " + str(fmt), *args)


def get_files(path, pattern):
    files = []
    for path in pathlib.Path(path).rglob(pattern):
        files.append(path.as_posix())
        log(path.as_posix())
    return files
