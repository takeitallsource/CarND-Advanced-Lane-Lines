from src import calibration as cal

calibrated, camera_matrix, distortion_coefficients, _,  = cal.load_calibration()

if not calibrated:
    calibrated, camera_matrix, distortion_coefficients, _, = cal.calibrate()