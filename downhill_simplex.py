from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from copy import copy
import csv
### https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
# Demonstration of the downhill simplex algorithm

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
    return Z[yIndex][xIndex]

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

    #find centroid of points
    centroid = [sum(simplex_data[0])/size, sum(simplex_data[1])/size]

    #compute reflection point
    reflection = [ centroid[0] + alpha*(centroid[0] - simplex_data[0][2]),
                   centroid[1] + alpha*(centroid[1] - simplex_data[1][2]) ]
    for i, val in enumerate(reflection):
        if val > limit:
            reflection[i] = limit
    f_r = f(reflection)
    if f_r >= simplex_data[2][0] and f_r < simplex_data[2][1]:
        print("Reflected")
        simplex_data[:, 2] = reflection + [f_r]
        return

    #Find expansion point
    if f_r < simplex_data[2][0]:
        print("Expanded")
        expansion = [ centroid[0] + gamma*(reflection[0] - centroid[0]),
                      centroid[1] + gamma*(reflection[1] - centroid[1]) ]
        for i, val in enumerate(expansion):
            if val > limit:
                expansion[i] = limit
        print(expansion)
        f_e = f(expansion)
        if f_e < f_r:
            simplex_data[:, 2] = expansion + [f_e]
        else:
            simplex_data[:, 2] = reflection + [f_r]
        return

    #Contract
    else: #(f_r >= f(simplex_data[2][1])
        contraction = [centroid[0] + rho*(simplex_data[0][2] - centroid[0]),
                       centroid[1] + rho*(simplex_data[1][2] - centroid[1]) ]
        f_c = f(contraction)
        if f_c < simplex_data[2][2]:
            print("Contract")
            simplex_data[:,2] = contraction + [f_c]
            return
    print("Shrink")
    #Shrink
    bestP = simplex_data[0:2, 0]
    for i in range(1, size):
        newP = [bestP[0] + sigma*(simplex_data[0][i] - bestP[0]),
                bestP[1] + sigma*(simplex_data[1][i] - bestP[1])]
        simplex_data[:, i] = newP + [f(newP)]
        
def load_data(filename):
    data = {'WOB':[], 'RPM':[], 'ROP':[]}
    last_depth = 9959.0818
    with open(filename) as f:
        f.readline()
        reader = csv.DictReader(f)
        for row in reader:
            if row['Depth - Bit']:
                data['ROP'].append(float(row['Depth - Bit']) - last_depth)
                last_depth = float(row['Depth - Bit'])
            if row['Top Drive RPM']:
                data['RPM'].append(float(row['Top Drive RPM']))
            if row['Weight on Bit']:
                data['WOB'].append(row['Weight on Bit'])
    return data
            
        
data = load_data('data/X14_00X14-GRB #1_TIME1SEC_Run #3.csv')

fig = plt.figure()
ax = fig.gca()

#X, Y, Z = axes3d.get_test_data(0.05)

ax.plot(data['RPM'])
ax.plot(data['ROP'])
#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)

    
#simplex_data = np.array([x,y,z])
#ax.plot(simplex_data[0], simplex_data[1], simplex_data[2],'m', label='Simplex', lw=4)
#coefficients
alpha = 2 
gamma = 2
rho = 0.5
sigma = 0.5
limit = 30

#ax.set_xlabel('X')
#ax.set_xlim(-40, 40)
#ax.set_ylabel('Y')
#ax.set_ylim(-40, 40)
#ax.set_zlabel('Z')
#ax.set_zlim(-100, 100)

fig.canvas.mpl_connect('button_press_event',onclick)
plt.show()