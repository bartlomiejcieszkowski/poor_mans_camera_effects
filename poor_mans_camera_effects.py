#!/usr/bin/env python

import numpy
import cv2
import getopt
import sys
import os
import numpy as np
import pathlib
import threading
import queue

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

shortopts = "lvc:"
longopts = ['list', 'ls', 'verbose', 'capture=', 'hq', 'facedetect', 'classifier_path=']
verbose = False
capture_idx = 0
cascade_classifiers_paths = []
facedetectors_idx = 0
facedetect = False
cascade_classifiers = None


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
    #read, img = capture.read()
    #if not read:
    #    capture.release()
    #    capture = None
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
            print("{} - fail".format(idx))
        else:
            read, img = camera.read()
            if read:
                consecutive = 0
                if verbose:
                    camera_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                    camera_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    camera_fps = camera.get(cv2.CAP_PROP_FPS)
                    print("[{}] {} x {} @ {}fps".format(idx, camera_width, camera_height, camera_fps))
                cameras.append(idx)
            camera.release()
        idx += 1
    return cameras

def get_detectors(path, pattern):
    detectors = []
    for path in pathlib.Path(path).rglob(pattern):
        detectors.append(path.as_posix())
        print(path.as_posix())
    return detectors


def input_loop():
    print("Input thread started")
    global cascade_classifiers
    while True:
        text = input()
        if text == 'f':
            name = 'frontalface'
            classifier_paths = cascade_classifiers_paths[name]
            new_idx = (classifier_paths[0] + 1) % len(classifier_paths[1])
            print("New classifier: {}".format(cascade_classifiers_paths[facedetectors_idx]))
            cascade_classifiers[name] = cv2.CascadeClassifier(classifier_paths[new_idx])
            classifier_paths[0] = new_idx
            cascade_classifiers_paths[name] = classifier_paths
        elif text == 'g':
            name = 'profileface'
            classifier_paths = cascade_classifiers_paths[name]
            new_idx = (classifier_paths[0] + 1) % len(classifier_paths[1])
            print("New classifier: {}".format(cascade_classifiers_paths[facedetectors_idx]))
            cascade_classifiers[name] = cv2.CascadeClassifier(classifier_paths[new_idx])
            classifier_paths[0] = new_idx
            cascade_classifiers_paths[name] = classifier_paths
        if text == 'h':
            name = 'smile'
            classifier_paths = cascade_classifiers_paths[name]
            new_idx = (classifier_paths[0] + 1) % len(classifier_paths[1])
            print("New classifier: {}".format(cascade_classifiers_paths[facedetectors_idx]))
            cascade_classifiers[name] = cv2.CascadeClassifier(classifier_paths[new_idx])
            classifier_paths[0] = new_idx
            cascade_classifiers_paths[name] = classifier_paths


def face_detect_fun(face_queue, bounding_boxes, scale_percent):
    print("Scale {}%".format(scale_percent))
    while True:
        frame = face_queue.get()
        print('New frame')
        detect_width = int(frame.shape[1] * scale_percent / 100)
        detect_height = int(frame.shape[0] * scale_percent / 100)
        dim = (detect_width, detect_height)
        frame_small = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        scaled_detections = cascade_classifiers['frontalface'].detectMultiScale(gray, 1.1, 4)
        if len(scaled_detections):
            detections = []
            for (x, y, w, h) in scaled_detections:
                detections.append((x * 100 // scale_percent, y * 100 // scale_percent, w * 100 // scale_percent, h * 100 // scale_percent))
            bounding_boxes[:] = detections
            if verbose:
                for (x, y, w, h) in bounding_boxes:
                    print("{}x{} {}x{} @ {}x{}".format(x, y, (x + w), (y + h), frame.shape[1], frame.shape[0]))




def usage():
    print("shortopts: {}".format(shortopts))
    print("longopts: {}".format(longopts))

def main():
    ls_mode = False
    force_hq = False
    classifier_path = os.getcwd()

    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts, longopts)
    except getopt.GetoptError as err:
        print(err)
        usage()
        return 2

    for o, a in opts:
        if o in ('-l', '--list', '--ls'):
            ls_mode = True
        elif o in ('-v', '--verbose'):
            global verbose
            verbose = True
        elif o in ('-c', '--capture'):
            global capture_idx
            try:
                capture_idx = int(a)
            except:
                print("{} - is not a valid index")
                return -1
        elif o in ('--hq'):
            force_hq = True
        elif o in ('--facedetect'):
            global facedetect
            facedetect = True
        elif o in('--classifier_path'):
            classifier_path = a
        else:
            assert False, 'unhandled option'

    if ls_mode:
        cameras = get_available_cameras()
        print(cameras)
        return 0

    if facedetect:
        global cascade_classifiers_paths
        global cascade_classifiers

        '*face*.xml'
        cascade_classifiers_paths = dict()
        cascade_classifiers_paths['frontalface'] = [0, get_detectors(classifier_path, '*frontalface*.xml')]
        cascade_classifiers_paths['profileface'] = [0, get_detectors(classifier_path, '*profileface*.xml')]
        cascade_classifiers_paths['smile'] = [0, get_detectors(classifier_path, '*smile*.xml')]

        cascade_classifiers = dict()
        if len(cascade_classifiers_paths['frontalface'][1]) == 0:
            cascade_classifiers['frontalface'] = None
        else:
            cascade_classifiers['frontalface'] = cv2.CascadeClassifier(cascade_classifiers_paths['frontalface'][1][facedetectors_idx])

        if len(cascade_classifiers_paths['profileface'][1]) == 0:
            cascade_classifiers['profileface'] = None
        else:
            cascade_classifiers['profileface'] = cv2.CascadeClassifier(cascade_classifiers_paths['profileface'][1][facedetectors_idx])

        if len(cascade_classifiers_paths['smile'][1]) == 0:
            cascade_classifiers['smile'] = None
        else:
            cascade_classifiers['smile'] = cv2.CascadeClassifier(cascade_classifiers_paths['profileface'][1][facedetectors_idx])

    # check if this is right
    camera = get_camera(capture_idx)
    if camera is None:
        print("Camera[{}] is unavailable".format(capture_idx))
        return -2

    if force_hq:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.set(cv2.CAP_PROP_FPS, 60)

    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = int(camera.get(cv2.CAP_PROP_FPS))
    if camera_fps == 0:
        camera_fps = 30
    print("{} x {} @ {}fps".format(camera_width, camera_height, camera_fps))

    # check if we get this fps

    virtual_camera_fps = camera_fps // 2
    virtual_camera = get_virtual_camera(camera_width, camera_height, virtual_camera_fps)

    # TEST_SECONDS = 5
    # test_frames = TEST_SECONDS * virtual_camera_fps
    # while test_frames > 0:
    #     read, frame = camera.read()
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     rgba_frame = np.zeros((camera_height, camera_width, 4), np.uint8)
    #     rgba_frame[:, :, :3] = rgb_frame
    #     rgba_frame[:, :, 3] = 255
    #     virtual_camera.send(rgba_frame)
    #     virtual_camera.sleep_until_next_frame()
    #     test_frames -= 1
    # fps = virtual_camera.current_fps
    # print(fps)
    # if (virtual_camera_fps - 5) > fps:
    #     virtual_camera_fps = virtual_camera_fps // 2
    #     virtual_camera.close()
    #     virtual_camera = None
    #
    #     virtual_camera = get_virtual_camera(camera_width, camera_height, virtual_camera_fps)
    #
    # print("settling on {}fps".format(virtual_camera_fps))

    rgba_frame = np.zeros((camera_height, camera_width, 4), np.uint8)
    bounding_boxes = []
    frame_idx = 0

    scale_percent = 25

    face_queue = queue.Queue()


    threads = []

    threads.append(threading.Thread(target=input_loop, name="Input"))
    if facedetect:
        threads.append(threading.Thread(target=face_detect_fun, args=(face_queue, bounding_boxes, scale_percent), name="Facedetect"))
        threads[-1].setDaemon(True)

    for thread in threads:
        print("Starting thread: \"{}\"".format(thread.getName()))

        thread.start()

    while True:
        read, frame = camera.read()
        if facedetect and frame_idx % (virtual_camera_fps * 5) == 0:
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_queue.put(frame)

        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (0, 0, 255), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgba_frame[:,:,:3] = rgb_frame
        rgba_frame[:,:,3] = 255
        virtual_camera.send(rgba_frame)
        # virtual_camera.sleep_until_next_frame()
        frame_idx += 1


    return 0


if __name__ == '__main__':
    sys.exit(main())
