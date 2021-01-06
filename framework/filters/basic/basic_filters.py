from abc import ABC

import cv2
import numpy as np
from scipy import interpolate

from framework.filters.filter_base import FilterBase


class Sharpen(FilterBase):
    def __init__(self):
        super().__init__()
        self.kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    def process(self, frame):
        return cv2.filter2D(frame, -1, self.kernel)


class GaussianBlur(FilterBase):
    def __init__(self):
        super().__init__()
        self.size = (35, 35)
        self.sigmaX = 0

    def process(self, frame):
        return cv2.GaussianBlur(frame, self.size, self.sigmaX)


class Blur(FilterBase):
    def __init__(self):
        super().__init__()
        self.ksize = (50, 50)

    def process(self, frame):
        return cv2.blur(frame, self.ksize)


class TemperatureBase(FilterBase, ABC):
    @staticmethod
    def spread_lookup_table(x, y):
        return interpolate.UnivariateSpline(x, y)(range(256))

    @staticmethod
    def increase_lut():
        return TemperatureBase.spread_lookup_table([0, 64, 128, 256], [0, 80, 160, 256])

    @staticmethod
    def decrease_lut():
        return TemperatureBase.spread_lookup_table([0, 64, 128, 256], [0, 50, 100, 256])


class Warm(TemperatureBase):
    def __init__(self):
        super().__init__()
        self.increase_lut = TemperatureBase.increase_lut()
        self.decrease_lut = TemperatureBase.decrease_lut()

    def process(self, frame):
        red_channel, green_channel, blue_channel = cv2.split(frame)
        red_channel = cv2.LUT(red_channel, self.increase_lookup_table).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, self.decrease_lookup_table).astype(np.uint8)
        return cv2.merge((red_channel, green_channel, blue_channel))


class Cold(TemperatureBase):
    def __init__(self):
        super().__init__()
        self.increase_lut = TemperatureBase.increase_lut()
        self.decrease_lut = TemperatureBase.decrease_lut()

    def process(self, frame):
        red_channel, green_channel, blue_channel = cv2.split(frame)
        red_channel = cv2.LUT(red_channel, self.decrease_lut).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, self.increase_lut).astype(np.uint8)
        return cv2.merge((red_channel, green_channel, blue_channel))


class ColorQuantization(FilterBase):
    def __init__(self):
        super().__init__()
        # max 20 iterations, epsilon
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        self.colors_num = 20

    def process(self, frame):
        data = np.float32(frame).reshape((-1, 3))

        ret, label, center = cv2.kmeans(data, self.colors_num, None, self.criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(frame.shape)
        return result

