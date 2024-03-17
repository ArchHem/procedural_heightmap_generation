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


def get_closest_grid(xmesh, ymesh, pos_vector):
    xrange = xmesh[0, :]
    yrange = ymesh[:, 0]

    yrange = yrange.reshape(1, len(yrange))
    xrange = xrange.reshape(1, len(xrange))

    x = pos_vector[:, 0:1]
    y = pos_vector[:, 1:2]

    dx = np.abs(xrange - x)
    dy = np.abs(yrange - y)

    id_x = np.argmin(dx, axis = 1)
    id_y = np.argmin(dy, axis = 1)

    return id_x, id_y


def not_in_texture(xmesh, ymesh, idx, idy):

    max_idx = xmesh.shape[1] - 2
    max_idy = ymesh.shape[0] - 2
    return np.where((idx >= max_idx) | (idy >= max_idy) | (idx <= 1) | (idy <= 1))


def get_accel(heightmap, posvec, velocvec, xmesh, ymesh, g, mu):

    #calculate best estimate for gradient

    xrange = xmesh[0, :]
    yrange = ymesh[:, 0]

    N_partics = posvec.shape[0]

    id_x, id_y = get_closest_grid(xmesh, ymesh, posvec)

    grad_x = (heightmap[id_y, id_x + 1] - heightmap[id_y, id_x-1]) / (xrange[id_x + 1] - xrange[id_x - 1])

    grad_y = (heightmap[id_y + 1, id_x] - heightmap[id_y - 1, id_x]) / (yrange[id_y + 1] - yrange[id_y - 1])

    normal_vector = np.zeros((len(grad_x),3))
    normal_vector[:,0] = -grad_x
    normal_vector[:,1] = -grad_y
    normal_vector[:,2] = 1

    normal_vector = normal_vector / np.sqrt(np.sum(normal_vector**2, axis = 1)).reshape(N_partics,1)

    n_force_x = g * normal_vector[:,0]
    n_force_y = g * normal_vector[:,1]

    f_force = g * normal_vector[:,2] * mu

    f_force_x = -velocvec[:,0] * f_force
    f_force_y = -velocvec[:,1] * f_force

    force_x = n_force_x + f_force_x
    force_y = n_force_y + f_force_y

    total_force = np.zeros((len(grad_x),2))
    total_force[:,0] = force_x
    total_force[:,1] = force_y

    indeces_to_stop = not_in_texture(xmesh, ymesh, id_x, id_y)

    return velocvec, total_force, indeces_to_stop


def batch_erosion(heightmap, xmesh, ymesh, dt = 0.05, N_particles = 100,
                        evap_rate = 0.05, g = 0.1, mu = 0.02, particle_volume = 0.02,
                        mtc = 0.02, tol = 1e-3, max_steps = 1000):

    maxx = xmesh[0,-2]
    maxy = ymesh[-2,0]

    minx = xmesh[0,2]
    miny = ymesh[2,0]

    x = np.random.uniform(minx,maxx,N_particles)
    y = np.random.uniform(miny,maxy,N_particles)

    posvec = np.zeros((N_particles,2))
    posvec[:,0] = x
    posvec[:,1] = y

    velvec = np.zeros((N_particles,2))

    og_vol = particle_volume
    truetol = tol * og_vol

    sediment_content = np.zeros(N_particles)

    for i in range(max_steps):

        particle_volume = (particle_volume)**(2/3) * (1 - evap_rate * dt)

        if particle_volume < truetol:
            break

        newvec, newaccel, termination = get_accel(heightmap,posvec,velvec,xmesh,ymesh, g, mu)

        potential_new_locs = posvec + dt * newvec

        checker_id_x, checker_id_y = get_closest_grid(xmesh, ymesh, potential_new_locs)
        to_nullify = not_in_texture(xmesh,ymesh,checker_id_x,checker_id_y)

        newvec[to_nullify] = 0.0
        newaccel[to_nullify] = 0.0

        posvec += newvec*dt
        velvec += newaccel*dt

        c_eq = particle_volume * np.sqrt(np.sum(velvec**2, axis = 1))

        sediment_content = mtc * (-sediment_content + c_eq)

        sediment_content[termination] = 0.0

        id_x, id_y = get_closest_grid(xmesh, ymesh, posvec)

        heightmap[id_y, id_x] -= dt * particle_volume * sediment_content

    return heightmap


def full_erosion(heightmap, xmesh, ymesh, N_batches = 2000, dt = 1.0, N_particles = 5,
                        evap_rate = 0.001, g = 1.0, mu = 0.05, particle_volume = 0.02,
                        mtc = 0.1, tol = 1e-2, max_steps = 1000):

    for j in tqdm(range(N_batches)):
        heightmap = batch_erosion(heightmap, xmesh, ymesh, dt = dt, N_particles = N_particles,
                        evap_rate = evap_rate, g = g, mu = mu, particle_volume = particle_volume,
                        mtc = mtc, tol = tol, max_steps = max_steps)

    return heightmap











