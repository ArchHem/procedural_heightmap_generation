import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from auxillary_functions import *

np.random.seed(2)

class base_terrain_generator:

    def __init__(self,size_x:int, size_y:int, scale_factor:float = 1.0, height_scaling:float = 60):
        """
        :param size_x: Number of pixels in x-axis
        :param size_y: Number of pixels in y-axis
        :param scale_factor: "Distance" that a pixel corresponds to
        generates:
        :param height_scaling: Highest possible point of the starting texture
        :xmesh, ymesh: stores meshgrid-like coordinates of each pixel
        :heightvalues: stores the 'height'. Internal methods assume it ranges from 0.0 to 1.0
        """
        xrange = np.array(range(0,size_x)) * scale_factor
        yrange = np.array(range(0,size_y)) * scale_factor

        xmesh, ymesh = np.meshgrid(xrange,yrange)

        heightvalues = np.zeros_like(xmesh)

        self.scale = scale_factor
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.height_scaling = height_scaling
        self.heightvalues = heightvalues

    def regenerate_perlin_heights(self,base_scale, seed = 0,
                                        N_octave = 5, persistence = 0.5,
                                        luna = 0.7):
        #TODO: Add docu here
        text = generate_multi_layered_perlin_noise(self.xmesh,self.ymesh, base_scale, seed = seed,
                                                   N_octave=N_octave, persistence=persistence, luna = luna)


        self.heightvalues = self.height_scaling * text

    def generate_perlin_heights(self,base_scale, seed = 0,
                                        N_octave = 5, persistence = 0.5,
                                        luna = 0.7):

        text = generate_multi_layered_perlin_noise(self.xmesh, self.ymesh, base_scale, seed=seed,
                                                   N_octave=N_octave, persistence=persistence, luna=luna)

        return text


    def regenerate_voronoi_heights(self,density,power,scaler):
        """

        :param density: Number of points per volume area to gnereate cells from
        :param power: what power we scale the exponential decay to
        :param scaler: how the exponent's decay is scaled linearly
        :return: updates self.heightvalues
        """

        z = generate_voronoi(density,power,scaler,self.xmesh,self.ymesh)

        self.heightvalues = z * self.height_scaling

    def generate_voronoi_heights(self,density, power, scaler):
        z = generate_voronoi(density, power, scaler, self.xmesh, self.ymesh)

        return z * self.height_scaling

    def add_tilt(self,xcomp:float = 0.0, ycomp:float = 0.0, local_scaler:float = 1.0):
        """

        :param xcomp:
        :param ycomp:
        :param local_scaler:
        :return:
        """

        normal_vector = np.array([xcomp,ycomp,1.0])
        normal_vector = normal_vector/np.sqrt(np.sum(normal_vector**2))

        plus_z = (normal_vector[0]*self.xmesh + normal_vector[1]*self.ymesh)/(-normal_vector[2])

        self.heightvalues = self.heightvalues + plus_z * self.height_scaling * local_scaler
        self.heightvalues = self.heightvalues


    def standard_eroder(self,N_partics = 8, N_batches = 10000, dt = 1.0,
                        evap_rate = 0.001, g = 1.0, mu = 0.2, particle_volume = 1.0,
                        mtc = 0.1, tol = 1e-2, max_steps = 1000):

        all_erosion(self.heightvalues, self.xmesh, self.ymesh, self.scale, N_batches = N_batches,
                        N_partics = N_partics, dt = dt,
                        evap_rate = evap_rate, g = g, mu = mu, particle_volume = particle_volume,
                        mtc = mtc, tol = tol, max_steps = max_steps)

    def regenerate_custom(self,array):

        xlen = array.shape[1]
        ylen = array.shape[0]

        xrange = np.array(range(0, xlen)) * self.scale
        yrange = np.array(range(0, ylen)) * self.scale

        xmesh, ymesh = np.meshgrid(xrange, yrange)

        self.xmesh = xmesh
        self.ymesh = ymesh
        self.heightvalues = array







test = base_terrain_generator(256,256,1.0)

test.regenerate_perlin_heights(64,N_octave=3, seed = 0)
test.add_tilt(0.0,0.000)
plt.imshow(test.heightvalues, cmap = 'gray')
z = test.heightvalues
fig, ax = plt.subplots()

test.standard_eroder(N_batches=20000, mtc = 0.1, evap_rate = 0.001, dt = 1.2, max_steps = 3000, mu = 0.1)
ax.imshow(test.heightvalues, cmap = 'gray')
plt.show()

