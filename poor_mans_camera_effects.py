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
import time
import click

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
longopts = ['list', 'ls', 'verbose', 'capture=', 'hq', 'facedetect', 'classifier_path=', 'yolo=']
verbose = False
capture_idx = 0
cascade_classifiers_paths = []
facedetectors_idx = 0
facedetect = False
cascade_classifiers = None
follow_face = True
interval_s = 5
filter_idx = 0
filters = [ None ]

color_map = {
    'face': (0, 0, 255),
    'smile': (255, 0, 0),
    'cat': (0, 255, 0)
}

"""
1. main
2. pars args
   a) list available cams - allow choosing cam that we will be using
3. passthrough cam to fakecam
"""

g_confidence = 0.9
g_threshold = 0.5

yolo_path = None

def log(fmt, *args):
    print(time.strftime("[%H:%M:%S] ", time.localtime())+fmt, *args)

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

def next_classifier(name):
    global cascade_classifiers
    classifier_paths = cascade_classifiers_paths[name]
    new_idx = (classifier_paths[0] + 1) % len(classifier_paths[1])
    print("{} - new classifier: {}".format(name, classifier_paths[1][new_idx]))
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
    print("interval: {}s".format(interval_s))

def change_filter(increment):
    global filter_idx
    filter_idx = (filter_idx + increment) % len(filters)

def input_loop():
    input_help = "f, g, h, j, t, i, o, h, ` "
    print("Input thread started")
    global interval_s
    global g_threshold
    global g_confidence
    input_lock = False
    while True:
        c = click.getchar()
        if c == '`':
              input_lock = not input_lock
              print("Input Lock? {}".format(input_lock))
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
                print(input_help)
            elif c == 'Q':
                exit(0)
            elif c == 'b':
                g_threshold -= 0.1
                print("g_threshold: {:.2f}".format(g_threshold))
            elif c == 'B':
                g_threshold += 0.1
                print("g_threshold: {:.2f}".format(g_threshold))
            elif c == 'n':
                g_confidence -= 0.1
                print("g_confidence: {:.2f}".format(g_confidence))
            elif c == 'N':
                g_confidence += 0.1
                print("g_confidence: {:.2f}".format(g_confidence))
            elif c == 'f':
                change_filter(-1)
            elif c == 'F':
                change_filter(1)
            else:
                print(c)



def yolo_detect(queue, bounding_boxes, yolo_path):
    print("Loading YOLO")
    net = cv2.dnn.readNetFromDarknet(os.path.join(yolo_path, "yolov3.cfg"), os.path.join(yolo_path, "yolov3.weights"))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    ln = net.getLayerNames()
    print("unconnected layers")
    for i in net.getUnconnectedOutLayers():
        print(i)
        print(ln[i[0] - 1])
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print(ln)

    while True:
        frame, frame_idx = queue.get()
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        log("YOLO processing took {:.6f} seconds".format(end - start))
        boxes = []
        confidences = []
        classIDs = []
        detections = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > g_confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, g_confidence, g_threshold)
        if len(idxs) > 0:
            for i in idxs.flatten():
                detections.append((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], i, confidences[i]))

        if len(detections):
            bounding_boxes[:] = detections


def face_detect_fun(face_queue, bounding_boxes, scale_percent, last_detect_idx):
    print("Scale {}%".format(scale_percent))
    while True:
        frame, frame_idx = face_queue.get()
        detect_width = int(frame.shape[1] * scale_percent / 100)
        detect_height = int(frame.shape[0] * scale_percent / 100)
        dim = (detect_width, detect_height)
        measurements = []
        measurements.append(time.time())
        frame_small = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        name = 'frontalface'
        scaled_detections = cascade_classifiers[name].detectMultiScale(gray, 1.1, 4)
        detections = []
        mirrored = False
        if len(scaled_detections) == 0:
            scaled_detections = cascade_classifiers['profileface'].detectMultiScale(gray, 1.1, 4)
            # try second profile
            if len(scaled_detections) == 0:
                scaled_detections = cascade_classifiers['profileface'].detectMultiScale(cv2.flip(gray, 1), 1.1, 4)
                mirrored = True
                if len(scaled_detections):
                    log("profileface - mirrored")
            else:
                log("profileface")
        else:
            log("frontalface")
        measurements.append(time.time())
        log("facedect - took {:.6f} seconds".format(measurements[-1] - measurements[0]))

        if len(scaled_detections):
            if mirrored:
                for (x, y, w, h) in scaled_detections:
                    detections.append((frame.shape[1] - (x * 100 // scale_percent),
                                      y * 100 // scale_percent,
                                      0 - (w * 100 // scale_percent),
                                      h * 100 // scale_percent,
                                      'face'))
            else:
                for (x, y, w, h) in scaled_detections:
                    detections.append((x * 100 // scale_percent, y * 100 // scale_percent, w * 100 // scale_percent, h * 100 // scale_percent, 'face'))
                # smile_detections = cascade_classifiers['smile'].detectMultiScale(gray[x:x+w, y:y+h], 1.1, 4)
                # for (sx, sy, sw, sh) in smile_detections:
                #     print("smile - {}x{} {}x{}", sx, sy, sx+sw, sy+sh)
                #     detections.append(((sx) * 100 // scale_percent, (sy) * 100 // scale_percent,
                #                        (sx+sw) * 100 // scale_percent, (sy+sh) * 100 // scale_percent, 'smile'))

        # scaled_detections = cascade_classifiers['cat'].detectMultiScale(gray, 1.05, minNeighbors=2)
        # if len(scaled_detections):
        #     for (x, y, w, h) in scaled_detections:
        #         detections.append((x * 100 // scale_percent, y * 100 // scale_percent, w * 100 // scale_percent,
        #                            h * 100 // scale_percent, 'cat'))

        if len(detections):
            last_detect_idx[0] = frame_idx
            bounding_boxes[:] = detections
            if verbose:
                for (x, y, w, h, name) in bounding_boxes:
                    print("[{}] {} {}x{} {}x{} @ {}x{}".format(frame_idx, name, x, y, (x + w), (y + h), frame.shape[1], frame.shape[0]))

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


def usage():
    print("shortopts: {}".format(shortopts))
    print("longopts: {}".format(longopts))

def main():
    ls_mode = False
    force_hq = False
    classifier_path = os.getcwd()
    global yolo_path
    global filters
    global filter_idx

    add_filters(filters)

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
        elif o in ('--classifier_path'):
            classifier_path = a
        elif o in ('--yolo'):
            yolo_path = a
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
        #cascade_classifiers_paths['smile'] = [0, get_detectors(classifier_path, '*smile*.xml')]
        #cascade_classifiers_paths['cat'] = [0, get_detectors(classifier_path, '*frontalcatface*.xml')]

        cascade_classifiers = dict()
        if len(cascade_classifiers_paths['frontalface'][1]) == 0:
            cascade_classifiers['frontalface'] = None
        else:
            cascade_classifiers['frontalface'] = cv2.CascadeClassifier(cascade_classifiers_paths['frontalface'][1][facedetectors_idx])

        if len(cascade_classifiers_paths['profileface'][1]) == 0:
            cascade_classifiers['profileface'] = None
        else:
            cascade_classifiers['profileface'] = cv2.CascadeClassifier(cascade_classifiers_paths['profileface'][1][facedetectors_idx])

        #if len(cascade_classifiers_paths['cat'][1]) == 0:
        #    cascade_classifiers['cat'] = None
        #else:
        #    cascade_classifiers['cat'] = cv2.CascadeClassifier(cascade_classifiers_paths['profileface'][1][facedetectors_idx])


        # if len(cascade_classifiers_paths['smile'][1]) == 0:
        #     cascade_classifiers['smile'] = None
        # else:
        #     cascade_classifiers['smile'] = cv2.CascadeClassifier(cascade_classifiers_paths['profileface'][1][facedetectors_idx])


    bounding_boxes_yolo = []
    bounding_boxes_face = []

    scale_percent = 50

    yolo_queue = queue.Queue()
    face_queue = queue.Queue()

    threads = []
    last_detect_idx = [ 0 ]

    threads.append(threading.Thread(target=input_loop, name="Input"))
    threads[-1].setDaemon(True)

    if facedetect:
        threads.append(threading.Thread(target=face_detect_fun, args=(face_queue, bounding_boxes_face, scale_percent, last_detect_idx), name="Facedetect"))
        threads[-1].setDaemon(True)

    if yolo_path:
        threads.append(threading.Thread(target=yolo_detect, args=(yolo_queue, bounding_boxes_yolo, yolo_path), name="YOLO"))
        threads[-1].setDaemon(True)

    threads.append(threading.Thread(target=frame_loop, args=(face_queue, yolo_queue, last_detect_idx, bounding_boxes_face, bounding_boxes_yolo), name="Frame processing"))
    threads[-1].setDaemon(True)

    for thread in threads:
        print("Starting thread: \"{}\"".format(thread.getName()))

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

def frame_loop(face_queue, yolo_queue, last_detect_idx, bounding_boxes_face, bounding_boxes_yolo):
    if yolo_path:
        LABELS = open(os.path.join(yolo_path, 'coco.names')).read().strip().split("\n")
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    frame_idx = 0
    camera = get_camera(capture_idx)
    if camera is None:
        print("Camera[{}] is unavailable".format(capture_idx))
        return -2

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FPS, 60)

    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = int(camera.get(cv2.CAP_PROP_FPS))
    if camera_fps == 0:
        camera_fps = 30
    print("{} x {} @ {}fps".format(camera_width, camera_height, camera_fps))

    virtual_camera_fps = camera_fps // 2
    virtual_camera = get_virtual_camera(camera_width, camera_height, virtual_camera_fps)


    auto_blur_delay_s = 5
    auto_blur_delay_frames = auto_blur_delay_s * virtual_camera_fps
    last_face_frame_idx = 0
    blur_pack = create_filter_blur2()

    rgba_frame = np.zeros((camera_height, camera_width, 4), np.uint8)
    rgba_frame[:, :, 3] = 255
    blur_count = 0

    while True:
        read, frame = camera.read()
        if facedetect and (interval_s == 0 or (frame_idx % (virtual_camera_fps * interval_s) == 0)):
            face_queue.put((frame, frame_idx))
        if yolo_path and (interval_s == 0 or (frame_idx % (virtual_camera_fps * interval_s) == 0)):
            yolo_queue.put((frame, frame_idx))

        #if follow_face and len(bounding_boxes):
        #    frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if filters[filter_idx]:
            filter = filters[filter_idx]
            log("Applying filter {}".format(filter[0]))
            frame = filter[1](frame, *filter[2])

        if (last_detect_idx[0] + auto_blur_delay_frames) < frame_idx:
            if blur_count == 0:
                log("auto blur")
            frame = blur_pack[1](frame, *blur_pack[2])
            blur_count += 1
        else:
            any_detection = False
            for (x, y, w, h, text) in bounding_boxes_face:
                any_detection = True
                show_detection(frame, x, y, x + w, y + h, color_map[text], text)
            for (x, y, w, h, classID, confidence) in bounding_boxes_yolo:
                any_detection = True
                color = [int(c) for c in COLORS[classID]]
                show_detection(frame, x, y, x + w, y + h, color, "{}: {:.4f}".format(LABELS[classID], confidence))
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
