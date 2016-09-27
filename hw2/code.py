import numpy as np
from numpy.linalg import eig

# Read in the points
points = np.loadtxt("cv_planar_points.txt", skiprows=1)

# Find the mean of all the points
r_g = points.mean(axis=0)

# and remove the mean
points_c = points - r_g

# compute the funky matrix S
S = np.zeros((3, 3))
for i in range(points_c.shape[0]):
    rc_i = points[i:i+1,:]
    S += np.dot(rc_i.T, rc_i)

# Take the minmal eigenvector
(w, v) = eig(S)
n = v[:, np.argmin(w)]

# finally compute d using the fi
d = n.T.dot(r_g)

# and print the result
print(n, d)
