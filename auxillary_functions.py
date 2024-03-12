import numpy as np
import numba as nb

np.random.seed(2)
@nb.njit()
def distance_exp(x, power = 2.0, scaler = 1.0):

    return np.exp(-np.abs(scaler) * x**power)


@nb.njit()
def generate_voronoi(density:float,power:float,scaler:float, xmesh, ymesh):

    #generate random positions in x-y

    ymax = np.amax(ymesh)
    xmax = np.amax(xmesh)

    total_area = ymax * xmax
    number_of_points = int(total_area * density)

    x_cords = np.random.uniform(0,xmax,number_of_points)
    y_cords = np.random.uniform(0,ymax,number_of_points)

    #face the might of numba - could do fancy indexing, but too much memory usage to expand into 3d for larger arrays
    output = np.zeros_like(ymesh)


    for x in range(xmesh.shape[0]):
        for y in range(xmesh.shape[1]):
            distances = np.sqrt((xmesh[x,y]-x_cords)**2 + (ymesh[x,y]-y_cords)**2)
            closest_distance = np.amin(distances)
            local_color = distance_exp(closest_distance,power,scaler)
            output[x,y] = local_color

    return output
@nb.njit()
def get_force(heightmap, x, y, xmesh, ymesh, mu):

    #calculate best estimate for gradient

    xrange = xmesh[0,:]
    yrange = ymesh[:,0]

    dx = np.abs(xrange - x)
    dy = np.abs(yrange - y)

    id_x = np.argmin(dx)
    id_y = np.argmin(dy)

    if id_x == 0 or id_y == 0 or id_x == len(dx) or id_y == len(dy):
        #catch edge cases
        return None

    else:
        #calculate 'gradient'

        grad_x = (heightmap[id_y, id_x + 1] - heightmap[id_y, id_x-1]) / (xrange[id_x + 1] - xrange[id_x - 1])
        grad_y = (heightmap[id_y + 1, id_x] - heightmap[id_y - 1, id_x]) / (yrange[id_y + 1] - yrange[id_y - 1])

        gradient = np.array([grad_x, grad_y])

        return gradient





@nb.njit()
def simple_drop_erosion(heightmap, xmesh, ymesh, time_steps, dt = 0.01, evap_rate = 0.02,
                        volume = 0.02, mass = 1.0, a_tol = 1e-4):

    pass


