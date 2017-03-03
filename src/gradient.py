import numpy as np
import cv2
import matplotlib.pyplot as plt


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
    # Grayscale
    # Apply cv2.Sobel()
    # Take the absolute value of the output from cv2.Sobel()
    # Scale the result to an 8-bit range (0-255)
    # Apply lower and upper thresholds
    # Create binary_output
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2) # g = sqrt(sx^2 + sy^2)

    scale_factor = np.max(gradient_magnitude) / 255

    gradient_magnitude = (gradient_magnitude / scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradient_magnitude)
    binary_output[(gradient_magnitude > thresh_min) & (gradient_magnitude < thresh_max)] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh_min=0., thresh_max = np.pi / 2):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)

    direction = np.arctan2(abs_sobel_y, abs_sobel_x)

    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh_min) & (direction < thresh_max)] = 1

    return binary_output


if __name__ == 'main':
    # sample = cv2.imread("data/signs_vehicles_xygrad.png")
    sample = cv2.imread("data/curved-lane.jpg")
    plt.imshow(abs_sobel_thresh(sample, thresh_min=20, thresh_max=100), cmap='gray')
    plt.waitforbuttonpress()

    plt.imshow(mag_thresh(sample, sobel_kernel=9, thresh_min=30, thresh_max=100), cmap='gray')
    plt.waitforbuttonpress()

    plt.imshow(dir_threshold(sample, sobel_kernel=15, thresh_min=0.7, thresh_max=1.3), cmap='gray')
    plt.waitforbuttonpress()

    # Trying combinations
    # Choose a Sobel kernel size
    ksize = 11 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(sample, orient='x', sobel_kernel=ksize, thresh_min=20, thresh_max=100)
    grady = abs_sobel_thresh(sample, orient='y', sobel_kernel=ksize, thresh_min=20, thresh_max=100)
    mag_binary = mag_thresh(sample, sobel_kernel=ksize, thresh_min=30, thresh_max=100)
    dir_binary = dir_threshold(sample, sobel_kernel=ksize, thresh_min=0.7, thresh_max=1.3)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    plt.imshow(combined, cmap='gray')
    plt.waitforbuttonpress()