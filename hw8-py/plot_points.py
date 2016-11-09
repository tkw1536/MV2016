import numpy as np
from matplotlib import pyplot as plt

# load the points
cv_circle_pts = np.loadtxt("pts_to_ransac.txt")



# make a cirlce
#circle = plt.Circle((2.042574e+00, -3.039281e+00), 3.142133e+00, color='red', fill=False)
fig, ax = plt.subplots()

ax = plt.gca()
ax.cla() # clear things for fresh plot


# plot the points
plt.plot(cv_circle_pts[:,0], cv_circle_pts[:,1], 'x')
# ax.add_artist(circle)

plt.show()
