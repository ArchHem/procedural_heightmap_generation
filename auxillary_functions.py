import numpy as np
import numba as nb

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
def get_force(heightmap, x, y, vx, vy, xmesh, ymesh, m, g, mu):

    #calculate best estimate for gradient

    xrange = xmesh[0,:]
    yrange = ymesh[:,0]

    dx = np.abs(xrange - x)
    dy = np.abs(yrange - y)

    id_x = np.argmin(dx)
    id_y = np.argmin(dy)

    if id_x == 0 or id_y == 0 or id_x == (len(dx) - 1) or id_y == (len(dy) - 1):
        #catch edge cases
        #will be used later
        arr = np.array([np.nan, np.nan])
        return arr

    else:
        #calculate 'gradient'

        grad_x = (heightmap[id_y, id_x + 1] - heightmap[id_y, id_x-1]) / (xrange[id_x + 1] - xrange[id_x - 1])

        grad_y = (heightmap[id_y + 1, id_x] - heightmap[id_y - 1, id_x]) / (yrange[id_y + 1] - yrange[id_y - 1])

        normal_vector = np.array([-grad_x, -grad_y, 1])
        normal_vector = normal_vector / np.sqrt(np.sum(normal_vector**2))

        n_force_x = m * g * normal_vector[0]
        n_force_y = m * g * normal_vector[1]

        f_force = m * g * normal_vector[2] * mu

        f_force_x = -vx * f_force
        f_force_y = -vy * f_force
        #get normal force
        force_x = n_force_x + f_force_x
        force_y = n_force_y + f_force_y

        return np.array([force_x,force_y])

@nb.njit()
def get_closest_grid(xmesh, ymesh, x, y):
    xrange = xmesh[0, :]
    yrange = ymesh[:, 0]

    dx = np.abs(xrange - x)
    dy = np.abs(yrange - y)

    id_x = np.argmin(dx)
    id_y = np.argmin(dy)

    return id_x, id_y








@nb.njit(fastmath = True, cache = True)
def simple_drop_erosion(heightmap, xmesh, ymesh, dt = 0.05,
                        evap_rate = 0.05, g = 0.1, mu = 0.02, particle_volume = 0.02,
                        initial_liquid_mass = 1.0, mtc = 0.0005, equil_concen = 0.5, tol = 1e-3):

    maxx = np.amax(xmesh)
    maxy = np.amax(ymesh)

    x = np.random.uniform(0,maxx)
    y = np.random.uniform(0,maxy)
    vx = 0.0
    vy = 0.0
    m_liquid = initial_liquid_mass
    m_sedi = 0.0
    m = m_sedi + m_liquid

    while True:
        lforce = get_force(heightmap,x,y,vx,vy,xmesh,ymesh,m,g,mu)
        if lforce[0] == np.nan:

            return heightmap

        else:
            #get sediment movememt

            delta_concetration = -mtc * (m_sedi/(m) - equil_concen) * dt

            d_sediment_mass = delta_concetration * particle_volume * (m)**2 / m_liquid

            m_sedi = m_sedi + d_sediment_mass
            m_liquid = m_liquid * (1 - evap_rate * dt)

            if m_liquid < tol:
                #if drop evaporated, deposit
                cix, ciy = get_closest_grid(xmesh, ymesh, x, y)


                heightmap[ciy, cix] = heightmap[ciy, cix] + m_sedi
                return heightmap

            m = m_sedi + m_liquid

            ax, ay = lforce[0] / m, lforce[1] / m

        x = x + vx
        y = y + vy

        vx = vx + ax
        vy = vy + ay

        cix, ciy = get_closest_grid(xmesh,ymesh, x, y)

        heightmap[ciy, cix] = heightmap[ciy, cix] - d_sediment_mass

@nb.njit(fastmath = True, cache = True)
def multi_drop_erosion(heightmap, xmesh, ymesh, dt = 0.05, number_of_particles = 1000,
                        evap_rate = 0.03, g = 0.5, mu = 0.02, particle_volume = 0.2,
                        initial_liquid_mass = 1.0, mtc = 0.05, equil_concen = 0.5, tol = 1e-3):

    for i in range(number_of_particles):
        heightmap = simple_drop_erosion(heightmap, xmesh, ymesh, dt = dt,
                        evap_rate = evap_rate, g = g, mu = mu, particle_volume = particle_volume,
                        initial_liquid_mass = initial_liquid_mass, mtc = mtc,
                                        equil_concen = equil_concen, tol = tol)

        print(i/number_of_particles)


    return heightmap


