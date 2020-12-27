import abc
import queue

from framework.base import log


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
    def thread_fun(detector):
        detector.main()

    @abc.abstractmethod
    def main(self):
        pass
