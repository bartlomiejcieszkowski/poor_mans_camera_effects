from framework.base import log, get_files, log_verbose, TimeMeasurements
from framework.detectors.detector_base import DetectorBase

import cv2


class CascadeClassifierDetector(DetectorBase):
    def __init__(self):
        super().__init__()
        self.classifier_files = dict()
        self.classifiers = dict()
        self.keys = []
        self.colors = dict()
        self.scale = 50
        self.time_measurements = TimeMeasurements()

    def next_classifier(self, name):
        self.add_idx(name, 1)
        self.create_classifier(name)

    def get_idx(self, key):
        return self.classifier_files[key][0]

    def add_idx(self, key, val):
        self.classifier_files[key][0] = (self.classifier_files[key][0] + val) % len(self.classifier_files[key][1])

    def create_classifier(self, key):
        if len(self.classifier_files[key][1]):
            log("loading classifier {} from {}".format(key, self.classifier_files[key][1][self.get_idx(key)]))
            self.classifiers[key] = cv2.CascadeClassifier(self.classifier_files[key][1][self.get_idx(key)])
        else:
            self.classifiers[key] = None

    def add_key(self, key, color):
        self.keys.append(key)
        self.colors[key] = color

    def set_scale(self, scale):
        self.scale = scale

    def setup(self, path):
        for key in self.keys:
            self.classifier_files[key] = [0, get_files(path, '*' + key + '*.xml')]
            self.create_classifier(key)

    def process(self, frame, frame_idx):
        detect_width = int(frame.shape[1]) * self.scale // 100
        detect_height = int(frame.shape[0]) * self.scale // 100
        dim = (detect_width, detect_height)
        frame_small = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        self.time_measurements.mark("resize")
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        name = 'frontalface'
        scaled_detections = self.classifiers[name].detectMultiScale(gray, 1.1, 4)
        detections = []
        mirrored = False
        if len(scaled_detections) == 0:
            name = 'profileface'
            scaled_detections = self.classifiers[name].detectMultiScale(gray, 1.1, 4)
            # try second profile
            if len(scaled_detections) == 0:
                scaled_detections = self.classifiers[name].detectMultiScale(cv2.flip(gray, 1), 1.1, 4)
                mirrored = True
                if len(scaled_detections):
                    log("profileface - mirrored")
            else:
                log("profileface")
        else:
            log("frontalface")
        self.time_measurements.mark("detect")

        if len(scaled_detections):
            if mirrored:
                for (x, y, w, h) in scaled_detections:
                    detections.append((frame.shape[1] - (x * 100 // self.scale),
                                      y * 100 // self.scale,
                                      0 - (w * 100 // self.scale),
                                      h * 100 // self.scale,
                                      name, self.colors[name]))
            else:
                for (x, y, w, h) in scaled_detections:
                    detections.append((
                        x * 100 // self.scale,
                        y * 100 // self.scale,
                        w * 100 // self.scale,
                        h * 100 // self.scale,
                        name, self.colors[name]))

        if len(detections):
            self.detected(frame_idx)
            self.bounding_boxes = detections
            if log_verbose():
                for (x, y, w, h, name, color) in self.bounding_boxes:
                    log("[{}] {} {}x{} {}x{} @ {}x{}".format(
                        frame_idx, name, x, y, (x + w), (y + h), frame.shape[1], frame.shape[0]))
