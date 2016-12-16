import cv2
import numpy as np
from numpy import linalg as la
from scipy import signal as sig

from matplotlib import pyplot as plt

import sys

# Settings
_neighborhood_size = 5
_half_neighborhood_size = _neighborhood_size // 2
_gauss_sigma = 1
_kernel_size = 6 * _gauss_sigma + 1
_k = int((_kernel_size - 1) / 2)
_num_corners = 100
_corner_quality_level = 0.001
_corner_min_distance = 10

# read video
vid = cv2.VideoCapture()
vid.open("videos/cars_at_an_intersection_hirez.mp4")

# get props
fps = vid.get(cv2.CAP_PROP_FPS)
vWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
vHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = vid.get(cv2.CAP_PROP_FOURCC)

vidWriter = cv2.VideoWriter('cars_at_an_intersection_optical_flow.avi',
                            int(fourcc), 0.75 * fps, (vWidth, vHeight))

# HACK: Skip ahead one second
for _ in range(int(fps) * 10):
    vid.read()

# read the first matrix
current_mat = cv2.cvtColor(vid.read()[1], cv2.COLOR_BGR2GRAY)

def quiver(image, pts, vx, vy, color=1):
    """ Creates a quiver mask"""

    (width, height) = image.shape

    for c in range(pts.shape[0]):
        x = pts[c, 0, 0]
        y = pts[c, 0, 1]

        xe = int(x + vx[c, 0])
        ye = int(y + vy[c, 0])

        if x < 0 or y < 0 or x >= height or y >= width:
            continue

        if xe < 0 or ye < 0 or xe >= height or ye >= width:
            continue

        image = cv2.line(image, (x, y), (xe, ye), color)

    return image

# Make a kernel for partial derivatives
I = np.repeat([np.arange(_kernel_size)], _kernel_size, axis=0).T
J = I.T

# derivative w.r.t. x
gauss_kernel_x = -((J - _k) / (2 * np.pi * _gauss_sigma ** 3)) * np.exp(
    - ((I - _k) ** 2 + (J - _k) ** 2) / (2 * _gauss_sigma ** 2))

# derivative w.r.t. y
gauss_kernel_y = gauss_kernel_x.T

kernel = (1 / (2 * np.pi * (_gauss_sigma ** 2))) * np.exp(
            -((I - _k) ** 2 + (J - _k) ** 2) / (2 * _gauss_sigma ** 2))


while True:
    ret, next_frame = vid.read()

    if not ret:
        break

    # grab the next frame
    next_mat = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    (xs, ys) = next_mat.shape

    # Compute x and y derivatives for both images
    Dx_1 = sig.convolve2d(current_mat, gauss_kernel_x)  # not filter2")
    Dx_2 = sig.convolve2d(next_mat, gauss_kernel_x)

    Dy_1 = sig.convolve2d(current_mat, gauss_kernel_y)
    Dy_2 = sig.convolve2d(next_mat, gauss_kernel_y)

    Ix = (Dx_1 + Dx_2) / 2
    Iy = (Dy_1 + Dy_2) / 2

    # extract corners using Eigenvalues corner detector
    corners_current = cv2.goodFeaturesToTrack(current_mat,
                                              _num_corners,
                                              _corner_quality_level,
                                              _corner_min_distance,
                                              useHarrisDetector=False)

    if corners_current is None:
        corners_current = [[[]]]
    corners_current = np.array(corners_current).astype(int)

    # smooth the images
    current_mat_smoothed = sig.convolve2d(current_mat, kernel)
    next_mat_smoothed = sig.convolve2d(next_mat, kernel)

    # I_t the partial derivative w.r.t. time
    It = next_mat_smoothed - current_mat_smoothed

    v_x = np.zeros((corners_current.shape[0], 1))
    v_y = np.zeros((corners_current.shape[0], 1))

    for c in range(corners_current.shape[0]):
        A = np.zeros((2, 2))
        B = np.zeros((2, 1))

        i = corners_current[c, 0, 1]
        j = corners_current[c, 0, 0]

        for m in range(i - _half_neighborhood_size,
                       i + _half_neighborhood_size + 1):
            for n in range(j - _half_neighborhood_size,
                           j + _half_neighborhood_size + 1):

                if (m < 0) or m >= xs or n < 0 or n >= ys:
                    continue

                A[0, 0] += Ix[m, n] * Ix[m, n]
                A[0, 1] += Ix[m, n] * Iy[m, n]
                A[1, 0] += Ix[m, n] * Iy[m, n]
                A[1, 1] += Iy[m, n] * Iy[m, n]

                B[0, 0] += Ix[m, n] * It[m, n]
                B[1, 0] += Iy[m, n] * It[m, n]

        Ainv = la.pinv(A)
        result = Ainv.dot(B)

        v_x[c] = result[1, 0]
        v_y[c] = result[0, 0]

    image = quiver(current_mat, corners_current, v_x, v_y)
    frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    vidWriter.write(frame)

    sys.stdout.write("*")
    sys.stdout.flush()

    # cv2.imshow("frame", current_mat.astype(np.uint))
    # cv2.waitKey(1)

    current_mat = next_mat
