import abc
import queue

from framework.base import log, TimeMeasurements


class FrameState(object):
    def __init__(self):
        self.last_detect_idx = 0

    def set_detect_idx(self, detect_idx):
        self.last_detect_idx = detect_idx

    def get_detect_idx(self):
        return self.last_detect_idx


class DetectorBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.input = queue.Queue(maxsize=1)
        self.bounding_boxes = []
        self.frame_state = None
        self.time_measurements = TimeMeasurements()

    def set_frame_state(self, frame_state: FrameState):
        self.frame_state = frame_state

    def detected(self, idx):
        log("idx: {}".format(idx))
        if self.frame_state is not None:
            log("updating")
            self.frame_state.set_detect_idx(idx)

    def get_bounding_boxes(self):
        return self.bounding_boxes

    def put(self, item):
        try:
            self.input.put(item, block=False)
        except queue.Full:
            log("queue full")

    @staticmethod
    def main_(detector):
        detector.main()

    def main(self):
        while True:
            frame, frame_idx = self.input.get()
            self.time_measurements.restart()
            self.process(frame, frame_idx)
            self.time_measurements.mark("end")
            self.time_measurements.log_total()

    @abc.abstractmethod
    def process(self, frame, frame_idx):
        pass