import numpy as np


def calculate_roi(image):
    shape = image.shape
    x_horizon = 0.1 * shape[0]  # overly fitted
    y_horizon = 0.1 * shape[1]  # overly fitted
    top_left = (shape[1] / 2 + y_horizon, shape[0] / 2 + x_horizon)
    top_right = (shape[1] / 2 - y_horizon, shape[0] / 2 + x_horizon)
    bottom_right = (y_horizon, shape[0])
    bottom_left = (shape[1] - y_horizon, shape[0])
    return np.array([[top_right, top_left, bottom_left, bottom_right]], dtype=np.int32)

def bird_eye(image):
    pass