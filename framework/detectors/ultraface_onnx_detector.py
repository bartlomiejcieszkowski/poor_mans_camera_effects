import cv2
import onnxruntime as ort
import numpy as np
import re

from framework.base import get_files, log
from framework.dependencies.box_utils import predict
from framework.dependencies.demo import scale
from framework.detectors.detector_base import DetectorBase


class UltrafaceOnnxDetector(DetectorBase):
    def __init__(self):
        super().__init__()
        self.models = None
        self.path = None
        self.detector = None
        self.threshold = 0.5
        self.color = (255, 128, 0)

    def setup(self, path):
        self.path = path
        self.models = [0, get_files(path, '*.onnx')]

    def get_current_model(self):
        return self.models[1][self.models[0]]

    @staticmethod
    def get_dimensions(path):
        log(path)
        width = int(re.match("^.*[-_](?P<width>[0-9]+)\\.onnx$", path).group('width'))
        return width, (width // 4) * 3

    def process(self, frame, frame_idx):
        model = self.get_current_model()
        log("Loading ONNX - {}".format(model))
        dimensions = self.get_dimensions(model)
        detector = ort.InferenceSession(model)
        input_name = detector.get_inputs()[0].name

        f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, dimensions)
        f_mean = np.array([127, 127, 127])
        f = (f - f_mean) / 128
        f = np.transpose(f, [2, 0, 1])
        f = np.expand_dims(f, axis=0)
        f = f.astype(np.float32)

        confidences, boxes = detector.run(None, {input_name: f})
        boxes, labels, probs = predict(frame.shape[1], frame.shape[0], confidences, boxes, self.threshold)

        detections = []
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            scale(boxes[i, :])
            detections.append((box[0], box[1], box[2] - box[0], box[3] - box[1],
                               "{}: {:.4f}".format(labels[i], probs[i]), self.color))

        if len(detections):
            self.detected(frame_idx)
            self.bounding_boxes = detections
