#!/usr/bin/env python

import numpy
import cv2

"""
1. main
2. pars args
   a) list available cams - allow choosing cam that we will be using
3. passthrough cam to fakecam
"""

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



def main():
    cameras = get_available_cameras()
    print(cameras)
    pass

if __name__ == '__main__':
    main()
