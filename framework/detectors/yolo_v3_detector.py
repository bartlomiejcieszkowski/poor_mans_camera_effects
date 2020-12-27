import os
import time

import cv2
import numpy as np

from framework.base import log
from framework.detectors.detector_base import DetectorBase


class YoloV3Detector(DetectorBase):
    def __init__(self):
        super().__init__()
        self.keys = []
        self.scale = 50
        self.path = None

        self.confidence = 0.9
        self.threshold = 0.5

    def setup(self, path):
        self.path = path



    def main(self):
        log("Loading YOLO")
        net = cv2.dnn.readNetFromDarknet(os.path.join(self.path, "yolov3.cfg"),
                                         os.path.join(self.path, "yolov3.weights"))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        ln = net.getLayerNames()
        log("unconnected layers")
        for i in net.getUnconnectedOutLayers():
            log(i)
            log(ln[i[0] - 1])
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        log(ln)

        LABELS = open(os.path.join(self.path, 'coco.names')).read().strip().split("\n")
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

        while True:
            frame, frame_idx = self.input.get()
            (H, W) = frame.shape[:2]

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

                    if confidence > self.confidence:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    color = [int(c) for c in COLORS[i]]
                    detections.append((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], "{}: {:.4f}".format(LABELS[i], confidences[i]), color))

            if len(detections):
                self.detected(frame_idx)
                self.bounding_boxes = detections
