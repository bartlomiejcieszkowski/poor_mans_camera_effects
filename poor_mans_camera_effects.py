#!/usr/bin/env python

import numpy
import cv2
import getopt
import sys
import os
import numpy as np

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

shortopts = "lvc:"
longopts = ['list', 'ls', 'verbose', 'capture=', 'hq', 'facedetect']
verbose = False
capture_idx = 0

def usage():
    print("shortopts: {}".format(shortopts))
    print("longopts: {}".format(longopts))

def main():
    ls_mode = False
    force_hq = False
    facedetect = False

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
            facedetect = True
        else:
            assert False, 'unhandled option'

    if ls_mode:
        cameras = get_available_cameras()
        print(cameras)
        return 0

    if facedetect:
        face_cascade = cv2.CascadeClassifier('opencv/data/haarcascade_frontalface_default.xml')

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
    TEST_SECONDS = 5
    test_frames = TEST_SECONDS * virtual_camera_fps
    while test_frames > 0:
        read, frame = camera.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgba_frame = np.zeros((camera_height, camera_width, 4), np.uint8)
        rgba_frame[:, :, :3] = rgb_frame
        rgba_frame[:, :, 3] = 255
        virtual_camera.send(rgba_frame)
        virtual_camera.sleep_until_next_frame()
        test_frames -= 1
    fps = virtual_camera.current_fps
    print(fps)
    if (virtual_camera_fps - 5) > fps:
        virtual_camera_fps = virtual_camera_fps // 2
        virtual_camera.close()
        virtual_camera = None

        virtual_camera = get_virtual_camera(camera_width, camera_height, virtual_camera_fps)

    print("settling on {}fps".format(virtual_camera_fps))

    rgba_frame = np.zeros((camera_height, camera_width, 4), np.uint8)
    faces = []
    frame_idx = 0
    while True:
        read, frame = camera.read()
        if facedetect and frame_idx % (virtual_camera_fps * 5) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if verbose:
                for (x, y, w, h) in faces:
                    print("[{}] {},{} {},{}", frame_idx, x, y, x+w, y+h)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgba_frame[:,:,:3] = rgb_frame
        rgba_frame[:,:,3] = 255
        virtual_camera.send(rgba_frame)
        # virtual_camera.sleep_until_next_frame()
        frame_idx += 1


    return 0


if __name__ == '__main__':
    sys.exit(main())
