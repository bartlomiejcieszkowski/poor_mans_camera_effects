import abc
import time
import threading
import pathlib
import enum
from collections import deque


class LogLevel(enum.IntEnum):
    ERROR = 1
    WARNING = 2
    INFO = 3
    VERBOSE = 4

import sys
from pathlib import Path

__level = LogLevel.INFO
__log_file = sys.stdout
__log_file_base = None
__log_file_path = None
__log_idx = 0
__log_idx_max = 5
__log_file_size_max = 5 * 1024 * 1024

def set_log_level(level: LogLevel):
    global __level
    __level = level


def get_log_level() -> LogLevel:
    return __level


def log_verbose() -> bool:
    return __level >= LogLevel.VERBOSE

def log_file_next():
    global __log_file
    global __log_file_path
    global __log_idx
    if __log_file is not sys.stdout:
        __log_idx = (__log_idx + 1) % __log_idx_max
    log_file(__log_file_base)


def log_file(file_path):
    global __log_file
    global __log_file_path
    global __log_file_base
    if file_path is None:
        __log_file = sys.stdout
    else:
        __log_file_base = file_path
        __log_file_path = Path(__log_file_base +  '.{}'.format(__log_idx))
        __log_file = __log_file_path.open('w')


def log_flush():
    __log_file.flush()


def log(fmt, *args):
    # if __log_file is not sys.stdout:
    #     if __log_file_size_max < __log_file_path.stat().st_size:
    #         log_file_next()

    print(time.strftime("[%H:%M:%S]", time.localtime())
          + "[{:<10.10}] ".format(threading.current_thread().name) + str(fmt), *args, file=__log_file)
    log_flush()



def msg(fmt, *args):
    print(fmt, *args)


def get_files(path, pattern):
    files = []
    for path in pathlib.Path(path).rglob(pattern):
        files.append(path.as_posix())
        log(path.as_posix())
    return files


class Threadable(object, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @staticmethod
    def main_(this):
        this.main()

    @abc.abstractmethod
    def main(self):
        raise NotImplementedError

    def create_thread(self, name: str = None, daemon: bool = True) -> threading.Thread:
        return threading.Thread(
            target=self.main_, args=(self,),
            name=(name if name is not None else type(self).__name__),
            daemon=daemon)


class TimeMeasurements:
    class LogMode(enum.IntEnum):
        NONE = 0,
        HUMAN = 1,
        CSV = 2,

    LOG_HUMAN = "TOTAL {total:>12} ns, PROCESS {process:>12} ns, THREAD {thread:>12} ns"

    LOG_CSV = "{time_str},{total},{process},{thread}"
    LOG_CSV_TIME = "%Y-%m-%d %H:%M:%S"

    LOG_NONE = ""

    @staticmethod
    def no_log(fmt, *args):
        pass

    def log_header(self):
        self.print_method(self.format_string)

    def __init__(self, log_mode=LogMode.HUMAN):
        self.measurements = deque()
        self.measure_method = time.perf_counter_ns
        self.processing_time = time.process_time_ns
        self.thread_time = time.thread_time_ns
        self.log_method = TimeMeasurements.no_log
        self.log_fmt = TimeMeasurements.LOG_NONE
        self.log_mode = log_mode
        self.log_time = False
        self.log_time_fmt = TimeMeasurements.LOG_NONE
        self.mode(log_mode)

    def mode(self, log_mode: LogMode):
        if log_mode is TimeMeasurements.LogMode.HUMAN:
            self.log_method = log
            self.log_fmt = TimeMeasurements.LOG_HUMAN
            self.log_time = False
            self.log_time_fmt = TimeMeasurements.LOG_NONE
        elif log_mode is TimeMeasurements.LogMode.CSV:
            self.log_method = TimeMeasurements.no_log
            self.log_fmt = TimeMeasurements.LOG_CSV
            self.log_time = True
            self.log_time_fmt = TimeMeasurements.LOG_CSV_TIME
        else:
            self.log_mode = TimeMeasurements.LogMode.NONE
            self.log_fmt = TimeMeasurements.LOG_NONE
            self.log_method = TimeMeasurements.no_log
            self.log_time = False
            self.log_time_fmt = TimeMeasurements.LOG_NONE
            return

        self.log_mode = log_mode

    def mark(self, name=''):
        self.measurements.append((self.measure_method(), self.processing_time(), self.thread_time(), name))

    def elapsed(self, idx=-1):
        return self.measurements[idx][0] - self.measurements[idx-1][0]

    def total(self):
        return self.measurements[-1][0] - self.measurements[0][0]

    def total_processing(self):
        return self.measurements[-1][1] - self.measurements[0][1]

    def total_thread(self):
        return self.measurements[-1][2] - self.measurements[0][2]

    def reset(self):
        self.measurements.clear()

    def restart(self):
        self.reset()
        self.mark("start")

    def log_total(self):
        if self.log_time:
            self.log_method(self.log_fmt.format(time=time.strftime(self.log_fmt, time.localtime()), total=self.total(),
                                                process=self.total_processing(), thread=self.total_thread()))
        else:
            self.log_method(self.log_fmt.format(total=self.total(), process=self.total_processing(),
                                                thread=self.total_thread()))

    # def log_steps(self):
    #    for idx in range(1, len(self.measurements)):
    #        self.log_method("{} - took {:>12} ns".format(self.measurements[idx][3], self.elapsed(idx)))

    def log_all(self):
        # if log_verbose():
        #    self.log_steps()
        self.log_total()
