import cv2

from framework.base import log, log_verbose


def get_camera(idx, api):
    capture = cv2.VideoCapture(idx, apiPreference=api)
    if not capture.isOpened():
        return None
    return capture


LIMIT_CONSECUTIVE = 3


def get_available_cameras(api):
    idx = 0
    cameras = []

    consecutive = 0
    while consecutive < LIMIT_CONSECUTIVE:
        camera = cv2.VideoCapture(idx, apiPreference=api)
        if not camera.isOpened():
            consecutive += 1
            log("{} - fail".format(idx))
        else:
            read, img = camera.read()
            if read:
                consecutive = 0
                if log_verbose():
                    camera_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                    camera_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    camera_fps = camera.get(cv2.CAP_PROP_FPS)
                    log("[{}] {} x {} @ {}fps".format(idx, camera_width, camera_height, camera_fps))
                cameras.append(idx)
            camera.release()
        idx += 1
    return cameras
