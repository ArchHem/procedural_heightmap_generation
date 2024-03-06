import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from auxillary_functions import *

class base_terrain_generator:

    def __init__(self,size_x:int, size_y:int, scale_factor:float = 1.0):
        """
        :param size_x: Number of pixels in x-axis
        :param size_y: Number of pixels in y-axis
        :param scale_factor: "Distance" that a pixel corresponds to
        """
        xrange = np.array(range(0,size_x)) * scale_factor
        yrange = np.array(range(0,size_y)) * scale_factor

        xmesh, ymesh = np.meshgrid(xrange,yrange)

        heightvalues = np.zeros_like(xmesh)

        self.scale = scale_factor
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.heightvalues = heightvalues

    def regenerate_perlin_heights(self,):

        pass

    def regenerate_voronoi_heights(self,density,power,scaler):

        z = generate_voronoi(density,power,scaler,self.xmesh,self.ymesh)

        self.heightvalues = z




test = base_terrain_generator(20,10,1.0)

test.regenerate_voronoi_heights(0.1,2.0,0.5)

