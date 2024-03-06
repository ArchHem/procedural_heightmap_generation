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
        xrange = np.array(range(0,size_x), dtype="Float64") * scale_factor
        yrange = np.array(range(0,size_y), dtype="Float64") * scale_factor

        xmesh, ymesh = np.meshgrid(xrange,yrange)

        heightvalues = np.zeros_like(xmesh)

        self.scale = scale_factor
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.heightvalues = heightvalues

    def regenerate_perlin_heights(self,):

        pass

    def regenerate_voronoi_heights(self,degree):

        pass






