import abc
import os
import subprocess
import sys
from typing import Tuple

import cv2
import numpy
import numpy as np

from framework.base import log, Threadable
from framework.camera import get_camera
from framework.filters.basic.basic_filters import Blur
from framework.filters.filter_base import AssemblyLine


class VideoParams:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps


if os.name == 'nt':
    import pyvirtualcam

    def get_virtual_camera(video_params: VideoParams):
        virtual_camera = pyvirtualcam.Camera(video_params.width, video_params.height, video_params.fps, 0)
        # get rid of warning
        if virtual_camera is not None:
            virtual_camera._fps_warning_printed = True
        return virtual_camera

else:
    import pyfakewebcam
    # modprobe v4l2loopback devices=2 # will create two fake webcam devices

    def get_virtual_camera(video_params: VideoParams):
        # naive search
        LIMIT = 10
        idx = 0
        while idx < LIMIT:
            try:
                camera = pyfakewebcam.FakeWebcam('/dev/video{}'.format(idx), video_params.width, video_params.height)
                return camera
            except:
                pass
            idx += 1
        return None




class Input(Threadable, metaclass=abc.ABCMeta):
    def __init__(self, frame_state, filter_manager, open_player=False):
        self.frame_idx = 0
        self.filter_manager = filter_manager
        self.detector_interval_s = 2
        self.auto_blur_delay_s = 10
        self.detectors = []
        self.frame_state = frame_state
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

    @abc.abstractmethod
    def get_frame(self) -> (int, numpy.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def init_video(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_video_params(self) -> VideoParams:
        raise NotImplementedError

    def main(self):
        self.frame_idx = 0
        if self.init_video() is False:
            return -2
        input_params = self.get_video_params()

        log("input {}x{}@{}fps".format(input_params.width, input_params.height, input_params.fps))
        if input_params.fps == 0:
            camera_fps = 30

        output_fps = (30 if input_params.fps == 0 else input_params.fps) // 2
        output_params = VideoParams(input_params.width, input_params.height, output_fps)
        virtual_camera = get_virtual_camera(output_params)

        rgba_frame = np.zeros((input_params.height, input_params.width, 4), np.uint8)
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

        auto_blur_delay_frames = self.auto_blur_delay_s * output_params.fps
        detector_delay_frames = self.detector_interval_s * output_params.fps

        while True:
            read, frame = self.get_frame()
            if read is False:
                failed_read += 1
                if failed_read >= failed_read_limit:
                    log("Failed reads {} - breaking".format(failed_read))
                    break
                continue

            frame = pre_process.process(frame)

            if (self.detector_interval_s == 0) or (self.frame_idx % detector_delay_frames == 0):
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


class CameraInput(Input):
    def __init__(self, frame_state, filter_manager, capture_idx=0, capture_api=cv2.CAP_MSMF, capture_params=VideoParams(1920, 1080, 60), open_player=False):
        super().__init__(frame_state, filter_manager, open_player)
        # cv2.CAP_DSHOW
        # cv2.CAP_MSMF

        self.capture_idx = capture_idx
        self.capture_api = capture_api
        self.requested_params = capture_params
        self.capture_params = capture_params
        self.capture = None

    def get_frame(self) -> (int, numpy.ndarray):
        return self.capture.read()

    def init_video(self):
        if self.capture is None:
            log("Getting camera {}".format(self.capture_idx))
            capture = get_camera(self.capture_idx, self.capture_api)
            if capture is None:
                log("Camera[{}] is unavailable".format(self.capture_idx))
                return False

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_params.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_params.height)
            capture.set(cv2.CAP_PROP_FPS, self.requested_params.fps)

            capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture_fps = int(capture.get(cv2.CAP_PROP_FPS))

            self.capture_params = VideoParams(capture_width, capture_height, capture_fps)
            self.capture = capture
        return True

    def get_video_params(self) -> VideoParams:
        return self.actual_capture_params


class FileInput(Input):
    def __init__(self, frame_state, filter_manager, file_path, loop, open_player=False):
        super().__init__(frame_state, filter_manager, open_player)
        # cv2.CAP_DSHOW
        # cv2.CAP_MSMF
        self.file_path = file_path
        self.capture = None
        self.capture_params = None
        self.loop = loop

    def init_video(self):
        if self.capture is None:
            log("Getting video {}".format(self.file_path))
            capture = cv2.VideoCapture(self.file_path)
            if capture is None:
                log("unable to open [{}]".format(self.capture_idx))
                return False

            capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture_fps = int(capture.get(cv2.CAP_PROP_FPS))

            self.capture_params = VideoParams(capture_width, capture_height, capture_fps)
            self.capture = capture
        return True

    def get_video_params(self) -> VideoParams:
        return self.capture_params

    def get_frame(self) -> (int, numpy.ndarray):
        read, frame = self.capture.read()
        if (frame is False) and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            read, frame = self.capture.read()
        return read, frame





