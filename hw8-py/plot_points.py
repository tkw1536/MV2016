import numpy as np
import json
from matplotlib import pyplot as plt

# load the points
cv_circle_pts = np.loadtxt("pts_to_ransac.txt")

# load results
with open('results.json') as f:
    circle_results = json.load(f)

circles = []

# make a cirlce
for circle in circle_results:
    circles.append(
        plt.Circle((circle['model'][0][0], circle['model'][0][1]), circle['model'][1], color='red', fill=False)
    )

fig, ax = plt.subplots()

ax = plt.gca()
ax.cla() # clear things for fresh plot


# plot the points
plt.plot(cv_circle_pts[:,0], cv_circle_pts[:,1], 'x')

for circle in circles:
    ax.add_artist(circle)

plt.show()
