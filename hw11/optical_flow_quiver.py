"""

optical_flow_quiver.py

Adapted and optimised version based on the original Matlab code
"""

import cv2
import tqdm
import numpy as np
from numpy import linalg as la
from scipy import signal as sig, stats as sta

import sys

############
# CONFIG
############

# Sit
_gauss_sigma = 1  # Sigma the Gaussian Kernels
_kernel_size = 6 * _gauss_sigma + 1
_k = int((_kernel_size - 1) / 2)

# Settings for corner detection
_num_corners = 100  # (Maximal) number of corners to detect per frame
_corner_quality_level = 0.001  # Quality measure for corners to detect
_corner_min_distance = 10  # Minimal distance between detected corners

# Neighbourhood for direction detection
_neighborhood_size = 5
_half_neighborhood_size = _neighborhood_size // 2

# color for the quiver plot
_color = 1


def quiver(img, pts, vx, vy, color=1):
    """ Makes a quiver plot on top of an existing image """

    (width, height) = img.shape

    for c in range(pts.shape[0]):
        x = pts[c, 0, 0]
        y = pts[c, 0, 1]

        xe = int(x + vx[c, 0])
        ye = int(y + vy[c, 0])

        if x < 0 or y < 0 or x >= height or y >= width:
            continue

        if xe < 0 or ye < 0 or xe >= height or ye >= width:
            continue

        img = cv2.line(img, (x, y), (xe, ye), color)

    return img


def pad2d(img, width, fill=0):
    """ Pads a two-dimensional image with a constant boundary"""

    return np.lib.pad(img, (width, width), 'constant', constant_values=fill)


def skip_time(cap, n):
    """ Skips a given amount of seconds in a VideoCapture """

    for _ in range(int(cap.get(cv2.CAP_PROP_FPS)) * n):
        cap.read()

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


# weighting kernel (also a gaussian)
nsig = 3
kernlen = (2*_half_neighborhood_size + 1) ** 2
interval = (2*nsig+1.)/(kernlen)
x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)

w1d = np.diff(sta.norm.cdf(x))
w_raw = np.sqrt(np.outer(w1d, w1d))
_W = w_raw/w_raw.sum()
_W_T_W = _W.T.dot(_W)

def process_frame(current_mat, next_mat):
    """ Processes a single from from the original image"""

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

    # if we have no corners, return the original frame
    if corners_current is None:
        return current_mat

    corners_current = np.array(corners_current).astype(int)

    # smooth the images
    current_mat_smoothed = sig.convolve2d(current_mat, kernel)
    next_mat_smoothed = sig.convolve2d(next_mat, kernel)

    # I_t the partial derivative w.r.t. time
    It = next_mat_smoothed - current_mat_smoothed

    v_x = np.zeros((corners_current.shape[0], 1))
    v_y = np.zeros((corners_current.shape[0], 1))

    # pad zeros all around our derivatives
    Ix = pad2d(Ix, _half_neighborhood_size)
    Iy = pad2d(Iy, _half_neighborhood_size)
    It = pad2d(It, _half_neighborhood_size)

    for c in range(corners_current.shape[0]):
        i = corners_current[c, 0, 1]
        j = corners_current[c, 0, 0]

        Ixs = Ix[i:i + 2 * _half_neighborhood_size + 1,
              j:j + 2 * _half_neighborhood_size + 1].flatten()

        Iys = Iy[i:i + 2 * _half_neighborhood_size + 1,
              j:j + 2 * _half_neighborhood_size + 1].flatten()

        Its = It[i:i + 2 * _half_neighborhood_size + 1,
              j:j + 2 * _half_neighborhood_size + 1].flatten()

        # find A, b by just smartly stacking the above
        A = np.vstack([Ixs, Iys]).T
        b = np.hstack([Its]).T

        # we compute the previous As and Bs from the code
        A_code = A.T.dot(_W_T_W).dot(A)
        B_code = A.T.dot(_W_T_W).dot(b)

        # compute the actual result
        Ainv = la.pinv(A_code)
        result = Ainv.dot(B_code)

        # and add it to the velocity
        v_x[c] = result[1]
        v_y[c] = result[0]

    return quiver(current_mat, corners_current, v_x, v_y, _color)

def main():
    """ Main entry point """

    # read video
    vid = cv2.VideoCapture()
    vid.open("videos/cars_at_an_intersection_hirez.mp4")

    # get props
    fps = vid.get(cv2.CAP_PROP_FPS)
    vWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = vid.get(cv2.CAP_PROP_FOURCC)
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    vidWriter = cv2.VideoWriter('cars_at_an_intersection_optical_flow.avi',
                                int(fourcc), 0.75 * fps, (vWidth, vHeight))

    # read the first matrix
    current_mat = cv2.cvtColor(vid.read()[1], cv2.COLOR_BGR2GRAY)

    T = tqdm.tqdm(total=frames - 1)

    while True:
        ret, next_frame = vid.read()

        if not ret:
            break

        # grab the next frame
        next_mat = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        image = process_frame(current_mat, next_mat)
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        vidWriter.write(frame)

        current_mat = next_mat
        T.update(1)

    T.close()


if __name__ == '__main__':
    main()
