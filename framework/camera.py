import cv2

from framework.base import log
from poor_mans_camera_effects import verbose


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
