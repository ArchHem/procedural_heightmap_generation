import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from auxillary_functions import *

class base_terrain_generator:

    def __init__(self,size_x:int, size_y:int, scale_factor:float = 1.0, height_scaling:float = 60):
        """
        :param size_x: Number of pixels in x-axis
        :param size_y: Number of pixels in y-axis
        :param scale_factor: "Distance" that a pixel corresponds to
        generates:
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

    def regenerate_perlin_heights(self,scaler):
        #TODO!!!!!!
        pass

    def regenerate_voronoi_heights(self,density,power,scaler):

        z = generate_voronoi(density,power,scaler,self.xmesh,self.ymesh)

        self.heightvalues = z * self.height_scaling

    def add_tilt(self,xcomp:float = 0.0,ycomp:float = 0.0):

        normal_vector = np.array([xcomp,ycomp,1.0])
        normal_vector = normal_vector/np.sqrt(np.sum(normal_vector**2))

        plus_z = (normal_vector[0]*self.xmesh + normal_vector[1]*self.ymesh)/(-normal_vector[2])

        self.heightvalues = self.heightvalues + plus_z
        self.heightvalues = self.heightvalues/np.amax(self.heightvalues) * self.height_scaling

    def standard_eroder(self):

        pass

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

test.regenerate_voronoi_heights(0.0001,1.5,0.001)
test.add_tilt(0.0,0.0)


y = all_erosion(test.heightvalues, test.xmesh, test.ymesh, test.scale)


plt.imshow(test.heightvalues, cmap = 'gray')
plt.show()

