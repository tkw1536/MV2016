import numpy as np
import json
from matplotlib import pyplot as plt
from ransac import run


points = np.loadtxt('pts_to_ransac.txt')

circle_results = run(points)

# A List of colors to use for the circles
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Circles and points to be plotted
circles = []
points = []
errors = []

# prepare the circles, points and errors
for idx, circle in enumerate(circle_results):

    # Circle in the right color
    circles.append(
        plt.Circle((circle['model'][0][0], circle['model'][0][1]), circle['model'][1], color=colors[idx % len(colors)], fill=False)
    )

    # also store the points seperatly
    points.append(np.array(circle['points']))

    # add the errors
    errors.append(circle['error'])



# Prepare the plot
fig, ax = plt.subplots()
ax = plt.gca()
ax.cla()


# Iterate over everything we have to plot
for (idx, (pts, circle, error)) in enumerate(zip(points, circles, errors)):

    if error is not None:
        # Plot the points in the right color
        plt.plot(pts[:,0], pts[:,1], '%sx' % (colors[idx % len(colors)], ), label='Circle %s (%s points, error %s)' % (idx+1, pts.shape[0], error))

        # and add the artist of the circle
        ax.add_artist(circle)
    else:
        plt.plot(pts[:,0], pts[:,1], 'kx', label='unassigned (%s points)' % (pts.shape[0]))

# show the plot
plt.legend()
plt.show()
