#!/usr/bin/env python3

import json
import numpy as np
from numpy import linalg
import random
from math import sqrt

def RANSAC(Z, fit, error, check_if_better, N_min, i_max, tau, d):
    model_best = None
    error_best = None
    I_best = None

    for _ in range(i_max):
        # select a random index
        I = np.array(random.sample(range(Z.shape[0]), N_min))

        # fit a model
        (model, isSingular) = fit(Z[I])
        if isSingular:
            continue

        # add all agreeing points
        for j in filter(lambda i: i not in I, range(Z.shape[0])):
            if error(model, Z[j]) < tau:
                I = np.append(I, [j])

        if I.shape[0] > d:
            # make a better model
            (model, _) = fit(Z[I])
            e = np.sum([error(model, Z[j]) for j in I])

            if check_if_better(model, e,  I, model_best, error_best, I_best):
                model_best = model
                error_best = e
                I_best = I

    return model_best, error_best, I_best

def check_if_better_1(model, e,  I, model_best, error_best, I_best):
    return model_best is None or (e < error_best)

def check_if_better_2(model, e,  I, model_best, error_best, I_best):
    if model_best is None:
        return True

    return (e / I.shape[0]) < (error_best / I_best.shape[0])

def check_if_better_3(model, e,  I, model_best, error_best, I_best):
    return (model_best is None) or (I.shape[0] > I_best.shape[0])


def circle_fit(Z):
    x = Z[0,:]
    y = Z[1,:]

    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0

    try:
        uc, vc = linalg.solve(A, B)
    except:
        return (None, True)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # compute distances from center
    Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1      = np.mean(Ri_1)
    residu_1 = np.sum((Ri_1-R_1)**2)

    # return the model
    return (((xc_1, yc_1), np.abs(R_1)), False)

def circle_error(model, point):
    (x, y) = point
    ((x0, y0), r) = model

    return (np.sqrt((x - x0)**2 + (y - y0)**2) - r) ** 2


def main():
    points = np.loadtxt('pts_to_ransac.txt')

    results = []

    N_min = 3
    i_max = 50
    tau = 0.2
    d = 20

    while points.shape[0] > N_min:
        model, error, index =\
                    RANSAC(points, circle_fit, circle_error,
                    check_if_better_2, N_min, i_max, tau, d)

        if model is None:
            break

        print(model)
        print(error)
        print("{0} points left.".format(points.shape[0]))
        print()

        results.append({
            "model": model,
            "error": error
        })

        # Remove 'used' points from dataset
        points = np.delete(points, index, 0)


    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
