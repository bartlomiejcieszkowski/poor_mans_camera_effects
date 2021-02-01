import os
import subprocess
import sys

import cv2
import numpy as np

from framework.base import log, Threadable
from framework.camera import get_camera
from framework.filters.basic.basic_filters import Blur
from framework.filters.filter_base import AssemblyLine


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
                camera = pyfakewebcam.FakeWebcam('/dev/video{}'.format(idx), int(width), int(height))
                return camera
            except:
                pass
            idx += 1
        return None


class CameraInput(Threadable):
    def __init__(self, frame_state, filter_manager, capture_idx=0, capture_api=cv2.CAP_MSMF, capture_params=(1920, 1080, 60), open_player=False):
        # cv2.CAP_DSHOW
        # cv2.CAP_MSMF
        self.frame_idx = 0
        self.capture_idx = capture_idx
        self.capture_api = capture_api
        self.capture_params = capture_params
        self.detector_interval_s = 2
        self.auto_blur_delay_s = 10
        self.detectors = []
        self.frame_state = frame_state
        self.filter_manager = filter_manager
        self.open_player = open_player
        self.player_process = None

    @staticmethod
    def show_detection(frame, x, y, xw, yh, color, text):
        cv2.rectangle(frame, (x, y), (xw, yh), color, 2)
        cv2.putText(frame, text, (x, yh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def add_interval(self, val):
        interval_s = self.detector_interval_s + val
        if interval_s < 0:
            interval_s = 0
        log("interval: {}s".format(interval_s))
        self.detector_interval_s = interval_s

    def main(self):
        self.frame_idx = 0
        log("Getting camera {}".format(self.capture_idx))
        camera = get_camera(self.capture_idx, self.capture_api)
        if camera is None:
            log("Camera[{}] is unavailable".format(self.capture_idx))
            return -2

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_params[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_params[1])
        camera.set(cv2.CAP_PROP_FPS, self.capture_params[2])

        camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_fps = int(camera.get(cv2.CAP_PROP_FPS))
        if camera_fps == 0:
            camera_fps = 30
        log("{} x {} @ {}fps".format(camera_width, camera_height, camera_fps))

        virtual_camera_fps = camera_fps // 2
        virtual_camera = get_virtual_camera(camera_width, camera_height, virtual_camera_fps)
        # get rid of warning
        virtual_camera._fps_warning_printed = True

        auto_blur_delay_frames = self.auto_blur_delay_s * virtual_camera_fps

        rgba_frame = np.zeros((camera_height, camera_width, 4), np.uint8)
        rgba_frame[:, :, 3] = 255
        blur_count = 0

        blur_line = AssemblyLine(self.filter_manager)
        blur_line.add_filter("Blur")

        # filters here would be applied even before detectors
        pre_process = AssemblyLine(self.filter_manager)

        # filters here would be applied last
        post_process = AssemblyLine(self.filter_manager)
        # post_process.add_filter("ColorQuantization")
        post_process.add_filter("CannyEdgeDetection")

        failed_read = 0
        failed_read_limit = 100


        if self.open_player:
            execute_path = [ os.path.abspath("c:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe") ]
            execute_args = "dshow:// :dshow-vdev=OBS-Camera :dshow-adev= :live-caching=0".split()
            self.player_process = subprocess.Popen(execute_path + execute_args)

        log("pre_process: {}".format(pre_process.to_string()))
        log("post_process: {}".format(post_process.to_string()))

        while True:
            read, frame = camera.read()
            if read is False:
                failed_read += 1
                if failed_read >= failed_read_limit:
                    log("Failed reads {} - breaking".format(failed_read))
                    break
                continue

            frame = pre_process.process(frame)

            if self.detector_interval_s == 0 or (self.frame_idx % (virtual_camera_fps * self.detector_interval_s) == 0):
                for detector in self.detectors:
                    detector.put((frame, self.frame_idx))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.filter_manager.current_filter().process(frame)

            if (self.frame_state.get_detect_idx() + auto_blur_delay_frames) < self.frame_idx:
                if blur_count == 0:
                    log("auto blur")
                frame = blur_line.process(frame)
                blur_count += 1
            else:
                any_detection = False
                for detector in self.detectors:
                    for (x, y, w, h, text, color) in detector.get_bounding_boxes():
                        any_detection = True
                        self.show_detection(frame, x, y, x + w, y + h, color, text)

                if any_detection:
                    blur_count = 0

            frame = post_process.process(frame)

            rgba_frame[:, :, :3] = frame
            # rgba_frame[:,:,3] = 255
            virtual_camera.send(rgba_frame)
            # virtual_camera.sleep_until_next_frame()
            self.frame_idx += 1

        return 0
