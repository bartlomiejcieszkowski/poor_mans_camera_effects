import argparse
import os
import sys
import threading
import time
from typing import Text, NoReturn

import click
import cv2

from framework.base import set_log_level, LogLevel, msg, log, Threadable
from framework.camera import get_available_cameras
from framework.detectors.cascade_classifier_detector import CascadeClassifierDetector
from framework.detectors.detector_base import FrameState
from framework.detectors.ultraface_onnx_detector import UltrafaceOnnxDetector
from framework.detectors.yolo_v3_detector import YoloV3Detector
from framework.filters.basic.basic_filters import add_default_filters
from framework.filters.filter_base import FilterManager
from framework.gui import UiThread
from framework.input import CameraInput


class Application(object):
    def __init__(self, argv):
        ap = argparse.ArgumentParser(prog=sys.argv[0])
        ap.add_argument('-l', '--list', action='store_true', help='prints available capture devices')
        ap.add_argument('-v', '--verbose', action='store_true', help='verbose')
        ap.add_argument('-c', '--capture', default=0, type=int, help='capture device number')
        ap.add_argument('--hq', action='store_true', help='high quality (1920x1080@60)')
        ap.add_argument('--haar-cascades', default=None, type=str)
        ap.add_argument('--yolo', default=None, type=str)
        ap.add_argument('--onnx', default=None, type=str)

        self.args = ap.parse_args(argv[1:])
        self.frame_state = None

        self.filter_manager = None
        self.input_thread = None
        self.camera_input = None
        self.ui_thread = None

        self.yolo_detector = None
        self.cascade_detector = None
        self.onnx_detector = None
        self.threads = []
        self.detectors = []

    def run(self):
        if self.args.verbose:
            set_log_level(LogLevel.VERBOSE)

        if self.args.list:
            cameras = get_available_cameras(cv2.CAP_MSMF)
            msg("Available cameras")
            msg(cameras)
            return 0

        log(self.args)

        self.frame_state = FrameState()

        if self.args.yolo and os.path.isdir(self.args.yolo):
            self.yolo_detector = YoloV3Detector()
            self.yolo_detector.setup(self.args.yolo)
            self.detectors.append(self.yolo_detector)

        if self.args.haar_cascades and os.path.isdir(self.args.haar_cascades):
            self.cascade_detector = CascadeClassifierDetector()
            self.cascade_detector.add_key('frontalface', (0, 0, 255))
            self.cascade_detector.add_key('profileface', (0, 0, 255))
            # self.cascade_detector.add_key('smile', (255, 0, 0))
            # self.cascade_detector.add_key('cat', (0, 255, 0))
            self.cascade_detector.setup(self.args.haar_cascades)
            self.detectors.append(self.cascade_detector)

        if self.args.onnx and os.path.isdir(self.args.onnx):
            self.onnx_detector = UltrafaceOnnxDetector()
            self.onnx_detector.setup(self.args.onnx)
            self.detectors.append(self.onnx_detector)

        self.filter_manager = FilterManager()
        add_default_filters(self.filter_manager)

        self.input_thread = InputTread(self)

        self.threads.append(self.input_thread.create_thread())

        for detector in self.detectors:
            detector.set_frame_state(self.frame_state)
            self.threads.append(detector.create_thread())

        self.camera_input = CameraInput(frame_state=self.frame_state, filter_manager=self.filter_manager, capture_idx=self.args.capture, open_player=True)
        self.camera_input.detectors = self.detectors

        self.threads.append(self.camera_input.create_thread(name="FrameProcessing"))

        self.ui_thread = UiThread()
        self.threads.append(self.ui_thread.create_thread())

        for thread in self.threads:
            log("Starting thread: \"{}\"".format(thread.getName()))
            thread.start()

        # simple watchdog
        while True:
            time.sleep(5)
            any_dead = False
            for thread in self.threads:
                if not thread.is_alive():
                    log("Thread(\"{}\") - died".format(thread.getName()))
                    any_dead = True
            if any_dead:
                # respawn or die
                return -1


class InputTread(Threadable):
    def __init__(self, app: Application):
        super().__init__()
        self.app = app

    def main(self):
        input_help = "`, G, g, i, o, h, Q"
        log("Input thread started")
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
                    self.app.cascade_detector.next_classifier('frontalface')
                elif c == 'g':
                    self.app.cascade_detector.next_classifier('profileface')
                elif c == 'i':
                    self.app.camera_input.add_interval(-1)
                elif c == 'o':
                    self.app.camera_input.add_interval(1)
                elif c == 'h':
                    log(input_help)
                elif c == 'Q':
                    log("Exit")
                    exit(0)
                else:
                    log(c)