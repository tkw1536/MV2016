import numpy as np
from matplotlib import pyplot as plt

# load the points
cv_circle_pts = np.loadtxt("cv_circle_pts.txt")

# plot them
plt.plot(cv_circle_pts[:,0], cv_circle_pts[:,1], 'x')
plt.show()
