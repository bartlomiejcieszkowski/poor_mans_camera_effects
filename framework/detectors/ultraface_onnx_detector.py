import cv2
import onnxruntime as ort
import numpy as np
import re

from framework.base import get_files, log
from framework.detectors.detector_base import DetectorBase


class UltrafaceOnnxDectector(DetectorBase):
    def __init__(self):
        super().__init__()
        self.models = None
        self.scale = 50
        self.path = None
        self.detector = None

    def setup(self, path):
        self.path = path
        self.models = [0, get_files(path, '*.onnx')]

    @staticmethod
    def get_dimensions(path):
        width = int(re.match("^.*-(?P<width>[0-9]+)\.onnx$", path).group('width'))
        return width, (width // 4) * 3

    def main(self):
        log("Loading ONNX")
        dimensions = self.get_dimensions(self.path)
        self.detector = ort.InferenceSession(self.path)
        input_name = self.detector.get_inputs()[0].name

        while True:
            frame, frame_idx = self.input.get()
            f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, dimensions)
            f_mean = np.array([127, 127, 127])
            f = (f - f_mean) / 128
            f = np.transpose(f, [2, 0, 1])
            f = np.expand_dims(f, axis=0)
            f = f.astype(np.float32)

            confidences, boxes = self.detector.run(None, {input_name: f})
            boxes, labels, probs = pre

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layer_outputs = net.forward(ln)
            end = time.time()
            log("YOLO processing took {:.6f} seconds".format(end - start))
            boxes = []
            confidences = []
            class_ids = []
            detections = []
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > g_confidence:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, g_confidence, g_threshold)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    color = [int(c) for c in COLORS[i]]
                    detections.append((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], "{}: {:.4f}".format(LABELS[i], confidences[i]), color))

            if len(detections):
                self.detected(frame_idx)
                self.bounding_boxes = detections
