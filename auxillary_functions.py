import numpy as np
import numba as nb
from tqdm import tqdm


np.random.seed(2)
@nb.njit()
def distance_exp(x, power = 2.0, scaler = 1.0):

    return np.exp(-np.abs(scaler) * x**power)


@nb.njit(fastmath = True, cache = True)
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


    for x in nb.prange(xmesh.shape[0]):
        for y in range(xmesh.shape[1]):
            distances = np.sqrt((xmesh[x,y]-x_cords)**2 + (ymesh[x,y]-y_cords)**2)
            closest_distance = np.amin(distances)
            local_color = distance_exp(closest_distance,power,scaler)
            output[x,y] = local_color

    return output

@nb.njit(fastmath = True, cache = True)
def get_closest_grid(x, y, scale):

    id_x = int(x/scale)
    id_y = int(y/scale)

    return id_x, id_y

@nb.njit(fastmath = True, cache = True)
def not_in_texture(xmesh, ymesh, idx, idy):

    N_y, N_x = xmesh.shape

    if idx <= 0 or idy <= 0 or idx >= (N_x - 1) or idy >= (N_y - 1):
        return True
    else:
        return False


@nb.njit(fastmath = True, cache = True)
def get_accel(heightmap, x, y, vx, vy, g, mu, scale):

    #calculate best estimate for gradient

    id_x, id_y = get_closest_grid(x, y, scale)

    grad_x = (heightmap[id_y, id_x + 1] - heightmap[id_y, id_x-1]) / (2 * scale)

    grad_y = (heightmap[id_y + 1, id_x] - heightmap[id_y - 1, id_x]) / (2 * scale)

    normal_vector = np.zeros((3,))
    normal_vector[0] = -grad_x
    normal_vector[1] = -grad_y
    normal_vector[2] = 1

    normal_vector = normal_vector / np.sqrt(np.sum(normal_vector**2))

    n_force_x = g * normal_vector[0]
    n_force_y = g * normal_vector[1]

    f_force = g * normal_vector[2] * mu

    f_force_x = -vx * f_force
    f_force_y = -vy * f_force

    force_x = n_force_x + f_force_x
    force_y = n_force_y + f_force_y

    total_accel = np.zeros((3,))
    total_accel[0] = force_x
    total_accel[1] = force_y

    total_accel[2] = g * normal_vector[2]

    return total_accel

@nb.njit(fastmath = True, cache = True)
def simple_erosion(heightmap, xmesh, ymesh, scale, dt = 0.05,
                        evap_rate = 0.05, g = 0.1, mu = 0.02, particle_volume = 0.02,
                        mtc = 0.02, tol = 1e-3, max_steps = 1000):

    d_heights = np.zeros_like(heightmap)
    maxx = xmesh[0,-2]
    maxy = ymesh[-2,0]

    minx = xmesh[0,1]
    miny = ymesh[1,0]

    x = np.random.uniform(minx,maxx)
    y = np.random.uniform(miny,maxy)

    vx = 0.0
    vy = 0.0

    og_vol = particle_volume
    truetol = tol * og_vol

    sediment_content = 0.0

    for i in range(max_steps):

        particle_volume = (particle_volume)**(2/3) * (1 - evap_rate * dt)

        if particle_volume < truetol:
            break

        accel_vec = get_accel(heightmap, x, y, vx, vy, g, mu, scale)

        x += vx
        y += vy

        vx += accel_vec[0]
        vy += accel_vec[1]

        newidx, newidy = get_closest_grid(x,y,scale)
        if not_in_texture(xmesh, ymesh, newidx, newidy):
            #terminate edge casses
            break

        #set equil. concentration to be proportinal to speed and z-acceleration

        c_eq = particle_volume * np.sqrt(vx**2 + vy**2) * np.abs(accel_vec[2])
        #not exactly physical, but close approx
        sediment_content = mtc * (-sediment_content + c_eq)
        id_x, id_y = get_closest_grid(x, y, scale)
        d_heights[id_y, id_x] -= dt * particle_volume * sediment_content

    return d_heights

@nb.njit(fastmath = True, parallel = True, cache = True)
def batch_erosion(heightmap, xmesh, ymesh, scale, N_partics = 8, dt = 1.0,
                        evap_rate = 0.001, g = 1.0, mu = 0.05, particle_volume = 1.0,
                        mtc = 0.1, tol = 1e-2, max_steps = 1000):

    result = np.zeros_like(heightmap)
    for j in nb.prange(N_partics):
        result += simple_erosion(heightmap, xmesh, ymesh, scale, dt = dt,
                        evap_rate = evap_rate, g = g, mu = mu, particle_volume = particle_volume,
                        mtc = mtc, tol = tol, max_steps = max_steps)
    return result

@nb.njit(fastmath = True)
def all_erosion(heightmap, xmesh, ymesh, scale, N_partics = 8, N_batches = 10000, dt = 1.0,
                        evap_rate = 0.001, g = 1.0, mu = 0.05, particle_volume = 1.0,
                        mtc = 0.1, tol = 1e-2, max_steps = 1000):

    for n in range(N_batches):
        local_erosion = batch_erosion(heightmap, xmesh, ymesh, scale, N_partics = N_partics, dt = dt,
                        evap_rate = evap_rate, g = g, mu = mu, particle_volume = particle_volume,
                        mtc = mtc, tol = tol, max_steps = max_steps)
        heightmap += local_erosion
        print(n/N_batches)

    return heightmap















