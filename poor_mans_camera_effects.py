#!/usr/bin/env python

import numpy
import cv2
import getopt
import sys
import os

if os.name == 'nt':
    import pyvirtualcam

    def get_virtual_camera(width, height, fps):
        return pyvirtualcam.Camera(width, height, fps)
else:
    import pyfakewebcam
    # modprobe v4l2loopback devices=2 # will create two fake webcam devices
    def get_virtual_camera(width, height, fps):
        # naive search
        LIMIT = 10
        idx = 0
        while idx < LIMIT:
            try:
                camera =  pyfakewebcam.FakeWebcam('/dev/video{}'.format(idx), width, height)
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

def available_camera(idx):
    capture = cv2.VideoCapture(idx)
    if not capture.isOpened():
        return False
    read, img = capture.read()
    capture.release()
    return read


def get_available_cameras():
    idx = 0
    cameras = []
    LIMIT_CONSECUTIVE = 3
    consecutive = 0
    while consecutive < LIMIT_CONSECUTIVE:
        capture = cv2.VideoCapture(idx)
        if not capture.isOpened():
            consecutive += 1
            print("{} - fail".format(idx))
        else:
            read, img = capture.read()
            print(read)
            if read:
                consecutive = 0
                cameras.append(idx)
            capture.release()
        idx += 1
    return cameras

shortopts = "lvc:"
longopts = ['list', 'ls', 'verbose', 'capture=']
verbose = False
capture_idx = 0

def usage():
    print("shortopts: {}".format(shortopts))
    print("longopts: {}".format(longopts))

def main():

    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts, longopts)
    except getopt.GetoptError as err:
        print(err)
        usage()
        return 2

    for o, a in opts:
        if o in ('-l', '--list', '--ls'):
            cameras = get_available_cameras()
            print(cameras)
            return 0
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
        else:
            assert False, 'unhandled option'

    # check if this is right
    if not available_camera(capture_idx):
        print("Camera[{}] is unavailable".format(capture_idx))
        return -2



    return 0


if __name__ == '__main__':
    sys.exit(main())
