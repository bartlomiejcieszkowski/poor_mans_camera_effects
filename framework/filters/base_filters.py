import cv2
import numpy as np


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
