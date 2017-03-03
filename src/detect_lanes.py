import cv2
import numpy as np
from src import calibration as cal
from src import colors as col
from src import gradient as grd
import matplotlib.pyplot as plt

OFFSET_Y = 50
OFFSET_X = 50
BOTTOM_OFFSET_1 = 200
BOTTOM_OFFSET_2 = 300


def calculate_unwarp_points(image, calibration_center):
    max_x, max_y = image.shape[1], image.shape[0]
    center_x, center_y = calibration_center[0], calibration_center[1]

    source = np.float32([[(BOTTOM_OFFSET_1, max_y), (center_x - OFFSET_X, center_y + OFFSET_Y),
                        (center_x + OFFSET_X, center_y + OFFSET_Y), (max_x - BOTTOM_OFFSET_2, max_y)]])
    destination = np.float32([[(BOTTOM_OFFSET_1, max_y), (BOTTOM_OFFSET_1, 0),
                        (max_x - BOTTOM_OFFSET_2, 0), (max_x - BOTTOM_OFFSET_2, max_y)]])
    return source, destination


def bird_eye(image, transformation_matrix):
    return cv2.warpPerspective(image, transformation_matrix, (image.shape[1], image.shape[0]))


def apply_gradient_color(image):
    gradient_sample = grd.abs_sobel_thresh(image, thresh_min=50, thresh_max=100)
    color_sample = col.hls_select(image, (170, 255))

    combined_binary = np.zeros_like(gradient_sample)
    combined_binary[(gradient_sample == 1) | (color_sample == 1)] = 1

    return combined_binary

def detect_lanes():
    calibrated, camera_matrix, distortion_coefficients, _, _ = cal.load_calibration()

    if not calibrated:
        calibrated, camera_matrix, distortion_coefficients, _, _ = cal.calibrate()

    image = cv2.imread("../test_images/test1.jpg")
    image = apply_gradient_color(image)
    undistorted = cv2.undistort(image, camera_matrix, distortion_coefficients, None, camera_matrix)
    source, dest = calculate_unwarp_points(image, camera_matrix[:, 2])
    transform_matrix = cv2.getPerspectiveTransform(source, dest)
    plt.imshow(bird_eye(undistorted, transform_matrix), cmap='gray')
    plt.show()

detect_lanes()