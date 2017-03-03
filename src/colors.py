import numpy as np
import cv2
from matplotlib import pyplot as plt


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls[:, :, 2]

    binary_output = np.zeros_like(s)
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary_output

if __name__ == 'main':
    sample = cv2.imread("data/curved-lane.jpg")

    plt.imshow(hls_select(sample, (90,255)), cmap='gray')
    plt.waitforbuttonpress()
