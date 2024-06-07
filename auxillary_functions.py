import numpy as np
import numba as nb
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator

nb.njit(fastmath = True, paralell = True, cache = True)
def generate_simple_perlin_noise(xmesh, ymesh, noise_scale, seed = 0):

    np.random.seed(seed)

    maxx = np.amax(xmesh)
    maxy = np.amax(ymesh)

    N_x = int(np.ceil(maxx / noise_scale)) + 2
    N_y = int(np.ceil(maxy / noise_scale)) + 2



    lxrange, lyrange = np.arange(-1, N_x) * noise_scale, np.arange(-1, N_y) * noise_scale


    #generate random unit vectors

    gradient_angles = np.random.uniform(0,2*np.pi,(len(lyrange), len(lxrange)))

    grad_x = np.cos(gradient_angles)
    grad_y = np.sin(gradient_angles)

    midx, midy = (lxrange[1:] + lxrange[:-1])/2, (lyrange[1:] + lyrange[:-1])/2

    number_of_y, number_of_x = len(midy), len(midx)

    texture = np.zeros((number_of_y, number_of_x))

    #look into whether this can be done via einsum
    for i in range(number_of_y):
        for j in nb.prange(number_of_x):

            lx = midx[j]
            ly = midy[i]

            bx = int(np.floor(lx / noise_scale))
            ux = int(np.ceil(lx / noise_scale))

            by = int(np.floor(ly / noise_scale))
            uy = int(np.ceil(ly / noise_scale))

            dx1 = lx - bx * noise_scale
            dy1 = ly - by * noise_scale

            dx2 = lx - bx * noise_scale
            dy2 = -ly + uy * noise_scale

            dx3 = -lx + ux * noise_scale
            dy3 = ly - by * noise_scale

            dx4 = -lx + ux * noise_scale
            dy4 = -ly + uy * noise_scale

            inn1 = dx1 * grad_x[by, bx] + dy1 * grad_y[by, bx]
            inn2 = dx2 * grad_x[uy, bx] + dy2 * grad_y[uy, bx]
            inn3 = dx3 * grad_x[by, ux] + dy3 * grad_y[by, ux]
            inn4 = dx4 * grad_x[uy, ux] + dy4 * grad_y[uy, ux]

            summed = inn1 + inn2 + inn3 + inn4

            texture[i,j] = summed


    #normalize to 0-1
    texture -= np.min(texture)

    texture /= np.amax(texture)

    #construct interpolator

    interp = RegularGridInterpolator((midy, midx), texture, method='cubic')

    final_text = interp((ymesh,xmesh))

    return final_text

nb.njit(fastmath = True, paralell = True)
def generate_multi_layered_perlin_noise(xmesh, ymesh, base_scale, seed = 0,
                                        N_octave = 5, persistence = 0.7,
                                        luna = 0.5):

    text_placeholder = np.zeros_like(xmesh)

    for i in nb.prange(N_octave):
        text_placeholder += luna**i * generate_simple_perlin_noise(xmesh, ymesh,
                                                         noise_scale=base_scale * persistence**i,
                                                                   seed = seed)
    text_placeholder /= np.amax(text_placeholder)

    return text_placeholder

@nb.njit()
def distance_exp(x, power = 2.0, scaler = 1.0):

    return np.exp(-np.abs(scaler) * x**power)

@nb.njit(fastmath = True, cache = True)
def normalize(x):

    norm = np.linalg.norm(x)
    if norm == 0.0:
        return x
    else:
        y = x/norm
        return y

@nb.njit(cache = True, fastmath = True)
def ReLu(x):

    return 0.5*(np.abs(x) + x)

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
def get_normal(id_x, id_y, heightmap, scale):
    d = scale
    """horizonal"""
    f_x1 = (heightmap[id_y, id_x + 1]-heightmap[id_y, id_x - 1]) / (2 * d)
    """vertical"""
    f_y1 = (heightmap[id_y + 1, id_x]-heightmap[id_y - 1, id_x]) / (2 * d)

    """cross"""
    normal_1 = np.array([-f_x1, -f_y1, 1])
    normal_1 = normalize(normal_1)

    f_x2 = 1/(2*np.sqrt(2)*d) * (heightmap[id_y-1, id_x - 1] + heightmap[id_y+1, id_x - 1]
                        - heightmap[id_y-1, id_x + 1] - heightmap[id_y+1, id_x + 1])

    f_y2 = 1/(2*np.sqrt(2)*d) * (heightmap[id_y-1, id_x - 1] - heightmap[id_y+1, id_x + 1]
                          - heightmap[id_y+1, id_x - 1] + heightmap[id_y-1, id_x + 1])

    normal_2 = np.array([f_x2, f_y2, 1])
    normal_2 = normalize(normal_2)

    avg_normal = normalize((normal_1 + normal_2)/2)

    return avg_normal

@nb.njit(fastmath = True, cache = True)
def get_accel(x, y, veloc, heightmap, scale, g, mu):

    id_x, id_y = get_closest_grid(x,y,scale)

    v_x, v_y, v_z = veloc

    veloc_norm = normalize(veloc)

    normal = get_normal(id_x,id_y, heightmap, scale)

    n_z = normal[2]

    a = g * (n_z * (normal - mu * veloc_norm) - np.array([0.0,0.0,1.0]))
    a_x, a_y, a_z = a

    return v_x, v_y, a_x, a_y, a_z, id_x, id_y

@nb.njit(fastmath = True, cache = True)
def trace_single_drop(heightmap, xmesh, ymesh, scale, mass,
                      mu, g, evap_rate, mtc,
                      density, veloc_prop, min_mass_ratio,
                      dt, max_timesteps):

    truetol = min_mass_ratio * mass

    sediment_concentr = 0.0

    maxx = xmesh[0, -2]
    maxy = ymesh[-2, 0]

    minx = xmesh[0, 1]
    miny = ymesh[1, 0]

    x = np.random.uniform(minx, maxx)
    y = np.random.uniform(miny, maxy)
    vx, vy, vz = 0.0, 0.0, 0.0

    for t in range(max_timesteps):

        veloc = np.array([vx, vy, vz])
        vl_x, vl_y, al_x, al_y, al_z, id_x, id_y = get_accel(x, y, veloc, heightmap, scale, g, mu)

        if not_in_texture(xmesh, ymesh, id_x, id_y):
            break

        xmid, ymid = x + dt/2 * vl_x, y + dt/2 * vl_y

        id_x_mid, id_y_mid = get_closest_grid(xmid, ymid, scale)

        evap_ratio = evap_rate * mass ** (2 / 3)
        mass -= dt * evap_ratio
        volume = mass/density

        if mass < truetol:
            heightmap[id_y_mid, id_x_mid] += sediment_concentr * volume
            break

        c_eq = veloc_prop * np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

        cdiff = mtc * (-sediment_concentr + c_eq)
        sediment_concentr += cdiff * dt

        x += vx*dt
        y += vy*dt
        vx += al_x*dt
        vy += al_y*dt
        vz += al_z*dt

        # deposit sediment
        heightmap[id_y, id_x] -= dt * volume * cdiff


@nb.njit(fastmath = True, cache = True, parallel = True)
def batch_erosion(heightmap, xmesh, ymesh, scale, mass,
                      mu, g, evap_rate, mtc,
                      density, veloc_prop, min_mass_ratio,
                      dt, max_timesteps, N_partics = 10):


    for j in nb.prange(N_partics):
        trace_single_drop(heightmap, xmesh, ymesh, scale, mass,
                      mu, g, evap_rate, mtc,
                      density, veloc_prop, min_mass_ratio,
                      dt, max_timesteps)

@nb.njit(fastmath = True)

def all_erosion(heightmap, xmesh, ymesh, scale, mass = 1.0,
                      mu = 0.2, g = 1.0, evap_rate = 0.05, mtc = 0.1,
                      density = 50, veloc_prop = 0.2, min_mass_ratio = 1e-3,
                      dt = 0.5, max_timesteps = 6000, N_partics = 10, N_batches = 20000):

    for n in range(N_batches):
        batch_erosion(heightmap, xmesh, ymesh, scale, mass,
                          mu, g, evap_rate, mtc,
                          density, veloc_prop, min_mass_ratio,
                          dt, max_timesteps, N_partics=N_partics)
        print(n/N_batches)

    return heightmap






























