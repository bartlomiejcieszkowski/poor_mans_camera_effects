#!/usr/bin/env python

import argparse
import os
import sys
import threading
import time

import click
import cv2
import numpy as np

from framework.base import log, set_log_level, LogLevel, msg
from framework.camera import get_available_cameras, get_camera
from framework.detectors.cascade_classifier_detector import CascadeClassifierDetector
from framework.detectors.detector_base import FrameState
from framework.detectors.ultraface_onnx_detector import UltrafaceOnnxDectector
from framework.detectors.yolo_v3_detector import YoloV3Detector
from framework.filters.basic.basic_filters import Sharpen, Blur, GaussianBlur, Warm, Cold
from framework.filters.filter_base import FilterManager, AssemblyLine
from framework.input import CameraInput


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


def add_filters(filter_manager):
    filter_manager.add(Sharpen)
    filter_manager.add(Blur)
    filter_manager.add(GaussianBlur)
    filter_manager.add(Warm)
    filter_manager.add(Cold)


def main():
    ls_mode = False
    force_hq = False
    classifier_path = os.getcwd()
    global frame_filters
    global filter_idx

    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--list', action='store_true', help='prints available capture devices')
    ap.add_argument('-v', '--verbose', action='store_true', help='verbose')
    ap.add_argument('-c', '--capture', default=0, type=int, help='capture device number')
    ap.add_argument('--hq', action='store_true', help='high quality (1920x1080@60)')
    ap.add_argument('--haar-cascades', default=None, type=str)
    ap.add_argument('--yolo', default=None, type=str)
    ap.add_argument('--onnx', default=None, type=str)

    args = ap.parse_args()



    if args.verbose:
        set_log_level(LogLevel.VERBOSE)

    if args.list:
        cameras = get_available_cameras(cv2.CAP_MSMF)
        msg("Available cameras")
        msg(cameras)
        return 0

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

    if args.onnx and os.path.isdir(args.onnx):
        onnx_detector = UltrafaceOnnxDectector()
        onnx_detector.setup(args.onnx)
        detectors.append(onnx_detector)

    onnx = False
    if args.onnx and os.path.isdir(args.onnx):
        onnx = True

    frame_filters = FilterManager()
    add_filters(frame_filters)

    assembly_line = AssemblyLine(frame_filters)
    threads.append(threading.Thread(target=assembly_line.main_, args=(assembly_line,), name=type(assembly_line).__name__, daemon=True))
    threads.append(threading.Thread(target=input_loop, name="Input", daemon=True))

    for detector in detectors:
        detector.set_frame_state(frame_state)
        threads.append(threading.Thread(target=detector.main_, args=(detector,), name=type(detector).__name__, daemon=True))

    camera_input = CameraInput(frame_state, frame_filters)
    camera_input.detectors = detectors
    threads.append(threading.Thread(target=camera_input.main_, args=(camera_input, ), name="FrameProcessing", daemon=True))

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


if __name__ == '__main__':
    sys.exit(main())
