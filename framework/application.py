import argparse
import os
import sys
import threading
import time
from typing import Text, NoReturn

import click
import cv2

from framework.base import set_log_level, LogLevel, msg, log, Threadable, log_file
from framework.camera import get_available_cameras
from framework.detectors.cascade_classifier_detector import CascadeClassifierDetector
from framework.detectors.detector_base import FrameState
from framework.detectors.ultraface_onnx_detector import UltrafaceOnnxDetector
from framework.detectors.yolo_v3_detector import YoloV3Detector
from framework.filters.basic.basic_filters import add_default_filters
from framework.filters.filter_base import FilterManager
from framework.gui import UiThread
from framework.input import CameraInput, FileInput
from framework.tui import TuiThread


class Application(object):
    def __init__(self, argv):
        ap = argparse.ArgumentParser(prog=sys.argv[0])
        ap.add_argument('-l', '--list', action='store_true', help='prints available capture devices')
        ap.add_argument('-v', '--verbose', action='store_true', help='verbose')
        ap.add_argument('--file', default=None, type=str, help='use file as input')
        ap.add_argument('-c', '--capture', default=0, type=int, help='capture device number')
        ap.add_argument('--capture-msmf', action='store_true', help='uses MSMF instead of DSHOW for capture')
        ap.add_argument('--hq', action='store_true', help='high quality (1920x1080@60)')
        ap.add_argument('--haar-cascades', default=None, type=str)
        ap.add_argument('--yolo', default=None, type=str)
        ap.add_argument('--onnx', default=None, type=str)
        ap.add_argument('--log', default='pmce.log', type=str)
        ap.add_argument('--open-player', action='store_true', help='opens vlc with the stream')

        self.args = ap.parse_args(argv[1:])
        self.frame_state = None

        self.filter_manager = None
        self.input_thread = None
        self.input = None
        self.ui_thread = None

        self.yolo_detector = None
        self.cascade_detector = None
        self.onnx_detector = None
        self.threads = []
        self.detectors = []

    def run(self):
        if self.args.verbose:
            set_log_level(LogLevel.VERBOSE)

        log_file(self.args.log)

        camera_api = cv2.CAP_MSMF if self.args.capture_msmf else cv2.CAP_DSHOW

        if self.args.list:
            cameras = get_available_cameras(camera_api)
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

        self.input_thread = TuiThread()

        self.threads.append(self.input_thread.create_thread())

        for detector in self.detectors:
            detector.set_frame_state(self.frame_state)
            self.threads.append(detector.create_thread())

        open_player = self.args.open_player

        if self.args.file:
            self.input = FileInput(frame_state=self.frame_state, filter_manager=self.filter_manager, file_path=self.args.file, loop=True, open_player=open_player)
        else:
            self.input = CameraInput(frame_state=self.frame_state, filter_manager=self.filter_manager, capture_idx=self.args.capture, capture_api=camera_api, open_player=open_player)
        self.input.detectors = self.detectors

        self.threads.append(self.input.create_thread(name="FrameProcessing"))

        # TODO: enable ui thread once there is any ui
        # self.ui_thread = UiThread()
        # self.threads.append(self.ui_thread.create_thread())

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


