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
    """Utterly deranged way of defining local grid, but it works and faster than meshgrid manipulation in this case,
    since we only interested in pairwise quantities"""
    d = scale

    z11 = heightmap[id_y, id_x]
    z12 = heightmap[id_y, id_x+1]
    z22 = heightmap[id_y+1, id_x+1]
    z21 = heightmap[id_y + 1, id_x]
    z20 = heightmap[id_y + 1, id_x -1]
    z10 = heightmap[id_y, id_x -1]
    z00 = heightmap[id_y-1, id_x-1]
    z01 = heightmap[id_y-1, id_x]
    z02 = heightmap[id_y-1, id_x + 1]

    xdispl = np.array([0.0,d,d,0.0,-d,-d,-d,.0,d])
    ydispl = np.array([0.0,0.0,d,d,d,0.0,-d,-d,-d])
    zdispl = np.array([z11, z12, z22, z21, z20,z10,z00, z01, z02])
    zdispl -= z11
    
    lvectors = np.zeros((9,3))
    lvectors[:, 0] = xdispl
    lvectors[:, 1] = ydispl
    lvectors[:, 2] = zdispl

    """Visualizing the local indices, centered on [1,1] bellow
    
    storing order of x11, x12, x22, x21, x20, x10, x00, x01, x02
    """

    #calculate triangulated mesh' midpoints as seen bellow

    """
    [2,0]-[2,1]-[2,2]
      |  X  |  X  |
    [1,0]-[1,1]-[1,2]
      |  X  |  X  |
    [0,0]-[0,1]-[0,2]
    
    We need to interpolate the value of z the mid-points 'X'
    and overwrite the corer values with them.
    """

    lvectors[2, 0:2] /= 2.
    lvectors[4, 0:2] /= 2.
    lvectors[6, 0:2] /= 2.
    lvectors[8, 0:2] /= 2.

    lvectors[2, 2] = (lvectors[0, 2] + lvectors[1, 2] + lvectors[2, 2] + lvectors[3, 2]) / 4
    lvectors[4, 2] = (lvectors[0, 2] + lvectors[3, 2] + lvectors[4, 2] + lvectors[5, 2]) / 4
    lvectors[6, 2] = (lvectors[0, 2] + lvectors[5, 2] + lvectors[6, 2] + lvectors[7, 2]) / 4
    lvectors[8, 2] = (lvectors[0, 2] + lvectors[7, 2] + lvectors[8, 2] + lvectors[1, 2]) / 4

    #calculate local normal vectors at 00 vertex
    tot_normal = np.zeros((3))
    for i in range(1,8):
        vl = np.cross(lvectors[i],lvectors[i+1])
        vl = vl / np.linalg.norm(vl)
        tot_normal += vl

    vl = np.cross(lvectors[8],lvectors[1])
    vl = vl / np.linalg.norm(vl)
    tot_normal += vl
    tot_normal /= 8
    tot_normal /= np.linalg.norm(tot_normal)

    return tot_normal

@nb.njit(fastmath = True, cache = True)
def p1_get_accel(x, y, vx, vy, vz, scale, mass, g, mu, heightmap):
    
    id_x, id_y = get_closest_grid(x,y,scale)
    
    smooth_normal = get_normal(id_x, id_y, heightmap, scale)

    smooth_tangent = np.array([vx,vy,vz])

    norm = np.linalg.norm(smooth_tangent)

    #deal with the edge case of zero velocity

    if np.linalg.norm(smooth_tangent) != 0.0:
        smooth_tangent /= norm

    else:
        #execute a gram-schmidt procedure
        prelim = smooth_normal + np.array([1.0,0.0,0.0])
        prelim /= np.linalg.norm(prelim)

        smooth_tangent = prelim - np.sum(prelim*smooth_normal) * smooth_normal
        smooth_tangent /= np.linalg.norm(smooth_tangent)


    smooth_binormal = np.cross(smooth_normal,smooth_tangent)
    # this neglects teh curvature

    #take projections of teh force into tangental, nomral and binormal directions

    #For the normal direction, we have |F_n| - m * g * N_z = m |v|^2 / r_curvature
    #we assume that the RHS is negibaly small

    Fn_mag = smooth_normal[2] * mass * g
    a_n = 0.0
    a_t = (-mu * Fn_mag - mass * g * smooth_tangent[2])/mass
    a_b = (-mass * g * smooth_binormal[2])/mass

    lcord_acc = np.array([a_n, a_t, a_b])

    transform_matrix = np.zeros((3,3))
    transform_matrix[0] = smooth_normal
    transform_matrix[1] = smooth_tangent
    transform_matrix[2] = smooth_binormal

    inverse_transform = np.linalg.inv(transform_matrix)

    cart_accel = inverse_transform @ lcord_acc

    ax, ay, az = cart_accel

    return vx, vy, vz, ax, ay, az, id_x, id_y

@nb.njit(cache = True, fastmath = True)
def p1_single_path_eroder(heightmap, xmesh, ymesh, scale, min_mass_ratio = 1e-3, init_mass = 1.0,
                          g = 0.1, volume = 0.02, mu_fric = 0.2,
                          max_timesteps = 1000, dt = 0.01, mtc = 0.02, evap_rate = 0.05):

    truetol = min_mass_ratio * init_mass

    sediment_content = 0.0

    maxx = xmesh[0, -2]
    maxy = ymesh[-2, 0]

    minx = xmesh[0, 1]
    miny = ymesh[1, 0]

    x = np.random.uniform(minx, maxx)
    y = np.random.uniform(miny, maxy)
    vx, vy, vz = 0.0, 0.0, 0.0
    id_x, id_y = get_closest_grid(x,y,scale)
    mass = init_mass

    delta_erosion = np.zeros_like(heightmap)

    for t in range(max_timesteps):

        #move particle
        vx, vy, vz, ax, ay, az, id_x, id_y = p1_get_accel(x, y, vx, vy, vz, scale, mass, g, mu_fric, heightmap)
        # terminate edge cases
        if not_in_texture(xmesh, ymesh, id_x, id_y):


            break
        #mass is propotitonal to r^3, whereas the evaprtation rate is proptironal to r^2
        evap_ratio = evap_rate * mass**(2/3)
        mass -= dt*evap_ratio
        volume -= dt*evap_ratio

        if mass < truetol:
            delta_erosion[id_y, id_x] += sediment_content
            break

        #pick up/put down sedement

        c_eq = volume * np.sqrt(vx**2 + vy**2 + vz**2)

        cdiff = mtc * (-sediment_content + c_eq)
        sediment_content += cdiff * dt

        x += vx
        y += vz
        vx += ax
        vy += ay
        vz += az

        #deposit sediment
        delta_erosion[id_y, id_x] -= dt * volume * cdiff

    return delta_erosion

@nb.njit(fastmath = True, parallel = True)
def p1_batch_erosion(heightmap, xmesh, ymesh, scale, min_mass_ratio = 1e-3, init_mass = 1.0,
                     g = 0.1, volume = 0.02, mu_fric = 0.2,
                     max_timesteps = 1000, dt = 0.01, mtc = 0.02, evap_rate = 0.05, N_partics = 8):

    result = np.zeros_like(heightmap)
    for j in nb.prange(N_partics):
        result += p1_single_path_eroder(heightmap, xmesh, ymesh, scale, min_mass_ratio, init_mass,
                                        g, volume, mu_fric,
                                        max_timesteps, dt, mtc, evap_rate)
    return result

@nb.njit(fastmath = True)
def p1_all_erosion(heightmap, xmesh, ymesh, scale, min_mass_ratio = 1e-3, init_mass = 1.0,
                   g = 0.1, volume = 0.02, mu_fric = 0.2,
                   max_timesteps = 1000, dt = 0.01, mtc = 0.02, evap_rate = 0.05, N_partics = 8,
                   N_batches  = 10000):

    for n in range(N_batches):
        local_erosion = p1_batch_erosion(heightmap, xmesh, ymesh, scale, min_mass_ratio, init_mass,
                                         g, volume, mu_fric,
                                         max_timesteps, dt, mtc, evap_rate, N_partics)
        heightmap += local_erosion
        print(n/N_batches)

    return heightmap













