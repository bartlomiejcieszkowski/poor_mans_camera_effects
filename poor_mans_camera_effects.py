#!/usr/bin/env python

import argparse
import os
import sys
import threading
import time

import click

from framework.base import log

if os.name == 'nt':
    import pyvirtualcam

    def get_virtual_camera(width, height, fps):
        return pyvirtualcam.Camera(int(width), int(height), fps, 0)
else:
    import pyfakewebcam
    # modprobe v4l2loopback devices=2 # will create two fake webcam devices
    def get_virtual_camera(width, height, fps):
        # naive search
        LIMIT = 10
        idx = 0
        while idx < LIMIT:
            try:
                camera =  pyfakewebcam.FakeWebcam('/dev/video{}'.format(idx), int(width), int(height))
                return camera
            except:
                pass
            idx += 1
        return None


verbose = False
capture_idx = 0

follow_face = True
interval_s = 5
filter_idx = 0
frame_filters = [None]


"""
1. main
2. pars args
   a) list available cams - allow choosing cam that we will be using
3. passthrough cam to fakecam
"""



def get_camera(idx):
    capture = cv2.VideoCapture(idx)
    if not capture.isOpened():
        return None
    return capture


def get_available_cameras():
    idx = 0
    cameras = []
    LIMIT_CONSECUTIVE = 3
    consecutive = 0
    while consecutive < LIMIT_CONSECUTIVE:
        camera = cv2.VideoCapture(idx)
        if not camera.isOpened():
            consecutive += 1
            log("{} - fail".format(idx))
        else:
            read, img = camera.read()
            if read:
                consecutive = 0
                if verbose:
                    camera_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                    camera_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    camera_fps = camera.get(cv2.CAP_PROP_FPS)
                    log("[{}] {} x {} @ {}fps".format(idx, camera_width, camera_height, camera_fps))
                cameras.append(idx)
            camera.release()
        idx += 1
    return cameras


def next_classifier(name):
    global cascade_classifiers
    classifier_paths = cascade_classifiers_paths[name]
    new_idx = (classifier_paths[0] + 1) % len(classifier_paths[1])
    log("{} - new classifier: {}".format(name, classifier_paths[1][new_idx]))
    cascade_classifiers[name] = cv2.CascadeClassifier(classifier_paths[1][new_idx])
    classifier_paths[0] = new_idx
    cascade_classifiers_paths[name] = classifier_paths

def change_interval(change):
    global interval_s
    if change < 0:
        if (interval_s + change) < 0:
            interval_s = 0
        else:
            interval_s += change
    elif change > 0:
        interval_s += change
    log("interval: {}s".format(interval_s))

def change_filter(increment):
    global filter_idx
    filter_idx = (filter_idx + increment) % len(frame_filters)

def input_loop():
    input_help = "f, g, h, j, t, i, o, h, ` "
    log("Input thread started")
    global interval_s
    global g_threshold
    global g_confidence
    input_lock = False
    while True:
        c = click.getchar()
        if c == '`':
              input_lock = not input_lock
              log("Input Lock? {}".format(input_lock))
        else:
            if input_lock:
                pass
            elif c == 'G':
                next_classifier('frontalface')
            elif c == 'g':
                next_classifier('profileface')
            elif c == 'h':
                next_classifier('smile')
            elif c == 'H':
                next_classifier('cat')
            elif c == 't':
                global follow_face
                follow_face = not follow_face
            elif c == 'i':
                change_interval(-1)
            elif c == 'o':
                change_interval(1)
            elif c == 'h':
                log(input_help)
            elif c == 'Q':
                log("Exit")
                exit(0)
            elif c == 'b':
                g_threshold -= 0.1
                log("g_threshold: {:.2f}".format(g_threshold))
            elif c == 'B':
                g_threshold += 0.1
                log("g_threshold: {:.2f}".format(g_threshold))
            elif c == 'n':
                g_confidence -= 0.1
                log("g_confidence: {:.2f}".format(g_confidence))
            elif c == 'N':
                g_confidence += 0.1
                log("g_confidence: {:.2f}".format(g_confidence))
            elif c == 'f':
                change_filter(-1)
            elif c == 'F':
                change_filter(1)
            else:
                log(c)


def filter_sharpen(frame, kernel):
    return cv2.filter2D(frame, -1, kernel)

def create_filter_sharpen():
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return ('sharpen', filter_sharpen, (kernel,))

def filter_blur(frame, size, sigmaX):
    return cv2.GaussianBlur(frame, size, sigmaX)

def create_filter_blur():
    return ('blur', filter_blur, ((35, 35), 0))

def filter_blur2(frame, ksize):
    return cv2.blur(frame, ksize)

def create_filter_blur2():
    return ('blur2', filter_blur2, ((50, 50),))

import scipy.interpolate

def spread_lookup_table(x, y):
    return scipy.interpolate.UnivariateSpline(x, y)(range(256))

def filter_warm(frame, increase_lookup_table, decrease_lookup_table):
    red_channel, green_channel, blue_channel = cv2.split(frame)
    red_channel = cv2.LUT(red_channel, increase_lookup_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decrease_lookup_table).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))

def create_filter_warm():
    increase_lookup_table = spread_lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
    decrease_lookup_table = spread_lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
    return ('warm', filter_warm, (increase_lookup_table, decrease_lookup_table))

def filter_cold(frame, increase_lookup_table, decrease_lookup_table):
    red_channel, green_channel, blue_channel = cv2.split(frame)
    red_channel = cv2.LUT(red_channel, decrease_lookup_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increase_lookup_table).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))

def create_filter_cold():
    increase_lookup_table = spread_lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
    decrease_lookup_table = spread_lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
    return ('cold', filter_cold, (increase_lookup_table, decrease_lookup_table))


def add_filters(filters):
    filters.append(create_filter_sharpen())
    filters.append(create_filter_blur())
    filters.append(create_filter_warm())
    filters.append(create_filter_cold())



import cv2
import onnxruntime as ort
import numpy as np
import re

class UltrafaceOnnxDectector(DetectorBase):
    def __init__(self):
        super().__init__()
        self.models = None
        self.scale = 50
        self.path = None
        self.detector = None

    def setup(self, path):
        self.path = path
        self.models = [0, get_detectors(path, '*.onnx')]

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




def main():
    ls_mode = False
    force_hq = False
    classifier_path = os.getcwd()
    global frame_filters
    global filter_idx

    add_filters(frame_filters)

    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--list', action='store_true', help='prints available capture devices')
    ap.add_argument('-v', '--verbose', action='store_true', help='verbose')
    ap.add_argument('-c', '--capture', default=0, type=int, help='capture device number')
    ap.add_argument('--hq', action='store_true', help='high quality (1920x1080@60)')
    ap.add_argument('--haar-cascades', default=None, type=str)
    ap.add_argument('--yolo', default=None, type=str)
    ap.add_argument('--onnx', default=None, type=str)

    args = ap.parse_args()

    global verbose

    if args.list:
        cameras = get_available_cameras()
        log(cameras)
        return 0

    verbose = args.verbose
    capture_idx = args.capture
    force_hq = args.hq

    log(args)

    frame_state = FrameState()

    detectors = []
    threads = []

    if args.yolo and os.path.isdir(args.yolo):
        yolo_detector = YoloV3Detector()
        yolo_detector.setup(args.yolo)
        detectors.append(yolo_detector)

    if args.haar_cascades and os.path.isdir(args.haar_cascades):
        cascade_detector = CascadeClassifierDetector()
        cascade_detector.add_key('frontalface', (0, 0, 255))
        cascade_detector.add_key('profileface', (0, 0, 255))
        #cascade_detector.add_key('smile', (255, 0, 0))
        #cascade_detector.add_key('cat', (0, 255, 0))
        cascade_detector.setup(args.haar_cascades)
        detectors.append(cascade_detector)

    onnx = False
    if args.onnx and os.path.isdir(args.onnx):
        onnx = True

    threads.append(threading.Thread(target=input_loop, name="Input", daemon=True))

    for detector in detectors:
        detector.set_frame_state(frame_state)
        threads.append(threading.Thread(target=detector.thread_fun, args=(detector,), name=type(detector).__name__, daemon=True))

    threads.append(threading.Thread(target=frame_loop, args=(detectors, frame_state), name="FrameProcessing", daemon=True))

    for thread in threads:
        log("Starting thread: \"{}\"".format(thread.getName()))
        thread.start()

    # simple watchdog
    while True:
        time.sleep(5)
        any_dead = False
        for thread in threads:
            if not thread.is_alive():
                log("Thread(\"{}\") - died".format(thread.getName()))
                any_dead = True
        if any_dead:
            # respawn or die
            return -1


def frame_loop(detectors, frame_state):
    frame_idx = 0
    log("Getting camera {}".format(capture_idx))
    camera = get_camera(capture_idx)
    if camera is None:
        log("Camera[{}] is unavailable".format(capture_idx))
        return -2

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FPS, 60)

    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = int(camera.get(cv2.CAP_PROP_FPS))
    if camera_fps == 0:
        camera_fps = 30
    log("{} x {} @ {}fps".format(camera_width, camera_height, camera_fps))

    virtual_camera_fps = camera_fps // 2
    virtual_camera = get_virtual_camera(camera_width, camera_height, virtual_camera_fps)

    auto_blur_delay_s = 10
    auto_blur_delay_frames = auto_blur_delay_s * virtual_camera_fps
    blur_pack = create_filter_blur2()

    rgba_frame = np.zeros((camera_height, camera_width, 4), np.uint8)
    rgba_frame[:, :, 3] = 255
    blur_count = 0

    while True:
        read, frame = camera.read()
        if interval_s == 0 or (frame_idx % (virtual_camera_fps * interval_s) == 0):
            for detector in detectors:
                detector.put((frame, frame_idx))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_filters[filter_idx]:
            frame_filter = frame_filters[filter_idx]
            log("Applying filter {}".format(frame_filter[0]))
            frame = frame_filter[1](frame, *frame_filter[2])

        if (frame_state.get_detect_idx() + auto_blur_delay_frames) < frame_idx:
            if blur_count == 0:
                log("auto blur")
            frame = blur_pack[1](frame, *blur_pack[2])
            blur_count += 1
        else:
            any_detection = False
            for detector in detectors:
                for (x, y, w, h, text, color) in detector.get_bounding_boxes():
                    any_detection = True
                    show_detection(frame, x, y, x + w, y + h, color, text)

            if any_detection:
                blur_count = 0

        rgba_frame[:,:,:3] = frame
        # rgba_frame[:,:,3] = 255
        virtual_camera.send(rgba_frame)
        # virtual_camera.sleep_until_next_frame()
        frame_idx += 1

    return 0


def show_detection(frame, x, y, xw, yh, color, text):
    cv2.rectangle(frame, (x, y), (xw, yh), color, 2)
    cv2.putText(frame, text, (x, yh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


if __name__ == '__main__':
    sys.exit(main())
