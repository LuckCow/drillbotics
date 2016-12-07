from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from copy import copy

def f(point):
    #Find Z value from X,Y,Z meshed data
    #Would be analagous to test drilling at x, y to get z
    xIndex, yIndex = 0, 0
    for i in range(len(X[0])):
        if abs(X[0][i] - point[0]) < 0.25: #Rounding up/down instead of extrapolating
            xIndex = i
            break
    for i in range(len(Y)):
        if abs(Y[i][0] - point[1]) < 0.25:
            yIndex = i
            break
    return Z[xIndex][yIndex]

def onclick(event):
    if event.button == 2:
        downhillSimplexIteration(simplex_data)
        #simplex.set_data([-25.0, 0., 25.], [-10, 12, 10])
        #simplex.set_3d_properties([0,0,0])
        fig.canvas.draw()

def downhillSimplexIteration(simplex_data):
    #Sort Points
    size = len(simplex_data)
    for i in range(size):
        for j in range(size - 1):
            if simplex_data[2][j] < simplex_data[2][j+1]:
                tmp = copy(simplex_data[:, j+1])
                simplex_data[:, j+1] = copy(simplex_data[:, j])
                simplex_data[:, j] = tmp
    
    centroid = [sum(simplex_data[0])/size, sum(simplex_data[1])/size]
    reflection = [ centroid[0] + alpha*(centroid[0] + simplex_data[0][2]),
                   centroid[1] + alpha*(centroid[1] + simplex_data[1][2]) ]
    f_r = f(reflection)
    if f_r >= simplex_data[2][0] and f_r < simplex_data[2][1]:
        print("Reflected")
        points[2] = reflection
        simplex_data[:, 2] = reflection + [f_r]
        return
        
    
        
        
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)

x = [-25.0, 0., 5.]
y = [-20., 25., 0.5]
z = []
for i in range(len(x)):
    z.append(f([x[i], y[i]]))

simplex_data = np.array([x,y,z])
simplex = ax.plot(x, y, z,'m', label='Simplex', lw=4)[0] #Need list here
alpha = 0.2

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

fig.canvas.mpl_connect('button_press_event',onclick)
plt.show()