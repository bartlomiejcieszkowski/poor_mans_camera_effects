import time
import threading
import pathlib

def log(fmt, *args):
    print(time.strftime("[%H:%M:%S]", time.localtime()) + "[" + threading.current_thread().name + "] " + str(fmt), *args)


def get_detectors(path, pattern):
    detectors = []
    for path in pathlib.Path(path).rglob(pattern):
        detectors.append(path.as_posix())
        log(path.as_posix())
    return detectors