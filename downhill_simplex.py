from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from copy import copy
### https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

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
        print(simplex_data)
        ax.plot(simplex_data[0], simplex_data[1], simplex_data[2],'m', label='Simplex', lw=4) 
        fig.canvas.draw()

def downhillSimplexIteration(simplex_data):
    #Sort Points
    size = len(simplex_data)
    for i in range(size):
        for j in range(size - 1):
            if simplex_data[2][j] > simplex_data[2][j+1]:
                tmp = copy(simplex_data[:, j+1])
                simplex_data[:, j+1] = copy(simplex_data[:, j])
                simplex_data[:, j] = tmp

    centroid = [sum(simplex_data[0])/size, sum(simplex_data[1])/size]
    reflection = [ centroid[0] + alpha*(centroid[0] + simplex_data[0][2]),
                   centroid[1] + alpha*(centroid[1] + simplex_data[1][2]) ]
    f_r = f(reflection)
    if f_r >= simplex_data[2][0] and f_r < simplex_data[2][1]:
        print("Reflected")
        simplex_data[:, 2] = reflection + [f_r]
        return
    if f_r < simplex_data[2][0]:
        print("Expanded")
        expansion = [ centroid[0] + gamma*(reflection[0] - centroid[0]),
                      centroid[1] + gamma*(reflection[1] - centroid[1]) ]
        f_e = f(expansion)
        if f_e < f_r:
            simplex_data[:, 2] = expansion + [f_e]
        else:
            simplex_data[:, 2] = reflection + [f_r]
        return
    else: #(f_r >= f(simplex_data[2][1])
        contraction = [centroid[0] + rho*(simplex_data[0][2] - centroid[0]),
                       centroid[1] + rho*(simplex_data[1][2] - centroid[1]) ]
        f_c = f(contraction)
        if f_c < simplex_data[2][2]:
            print("Contract")
            simplex_data[:,2] = contraction + [f_c]
            return
    print("Shrink")
    bestP = simplex_data[0:2, 0]
    for i in range(1, size):
        newP = [bestP[0] + sigma*(simplex_data[0][i] - bestP[0]),
                bestP[1] + sigma*(simplex_data[1][i] - bestP[1])]
        simplex_data[:, i] = newP + [f(newP)]
        
        
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)

x = [20, -20., -10.]
y = [-20., 0, 0.5]
z = []
for i in range(len(x)):
    z.append(f([x[i], y[i]]))

print(f([-16,-6]), f([15,20]))
    
simplex_data = np.array([x,y,z])
ax.plot(simplex_data[0], simplex_data[1], simplex_data[2],'m', label='Simplex', lw=4) 
alpha = 1
gamma = 2
rho = 0.5
sigma = 0.5

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

fig.canvas.mpl_connect('button_press_event',onclick)
plt.show()