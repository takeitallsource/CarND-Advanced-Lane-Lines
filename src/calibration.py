import glob
import pickle
import os
import numpy as np
import cv2

CALIBRATION_FILENAME = "config/calibration.pickle"


def calibrate(source_glob="camera_cal/*.jpg", pattern_size=(9, 6), persist=True):
    real_world_grid = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    real_world_grid[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    real_world_points = []
    transformed_points = []

    calibration_images = glob.glob(source_glob)

    for idx, image_file in enumerate(calibration_images):
        chessboard_image = cv2.imread(image_file, flags=cv2.IMREAD_GRAYSCALE)
        found_corners, corners = cv2.findChessboardCorners(chessboard_image, pattern_size, None)
        if found_corners:
            real_world_points.append(real_world_grid)
            transformed_points.append(corners)

            image_size = (chessboard_image.shape[1], chessboard_image.shape[0])

            calibrated, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = \
                cv2.calibrateCamera(real_world_points, transformed_points, image_size, cameraMatrix=None, distCoeffs=None)

            if calibrated and persist:
                calibration_info = {
                    "mtx": camera_matrix,
                    "dst": distortion_coefficients,
                    "rot": rotation_vectors,
                    "tra": translation_vectors
                }
                with open(CALIBRATION_FILENAME, "+wb") as f:
                    pickle.dump(calibration_info, f)

            return calibrated, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors

    raise ValueError()


def load_calibration():
    if os.path.exists(CALIBRATION_FILENAME):
        with open(CALIBRATION_FILENAME, "rb") as f:
            calibration_info = pickle.load(f)
        return True, calibration_info["mtx"], calibration_info["dst"], calibration_info["rot"], calibration_info["tra"]
    else:
        return False, None, None, None, None
