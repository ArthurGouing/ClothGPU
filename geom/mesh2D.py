#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# This file is part of SimulationTeachingElan, a python code used for teaching at Elan Inria.
#
# Copyright 2020 Mickael Ly <mickael.ly@inria.fr> (Elan / Inria - Université Grenoble Alpes)
# SimulationTeachingElan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SimulationTeachingElan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SimulationTeachingElan.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
from .rod2D import Rod2D

#import pyassimp

## Class defining a 2D mesh
class Mesh2D(Rod2D):

    def __init__(self, positions, indices, colours = None):
        ## Constructor
        # @param positions  1-D Numpy array concatenating
        #                   the 2D positions of the mesh
        # @param indices    1-D Numpy array for the triangles indices (triplets)
        # @param colours    1-D Numpy array concatenating the
        #                   vertices colours (r, g, b)

        super().__init__(positions, colours)        
        self.indices = np.array(indices, np.int32)

        # Build neighbours list
        self.neighbours = [ set() for i in range(self.nbVertices)]
        for i in range(int(self.indices.size / 3)):
            t = self.indices[3 * i : 3 * i + 3]
            for j in range(3):
                self.neighbours[t[j]].add(t[(j + 1) % 3])
                self.neighbours[t[j]].add(t[(j + 2) % 3])
        for i in range(self.nbVertices):
            self.neighbours[i] = list(self.neighbours[i])
    

"""
def loadMeshes(filename):
    ## Loader function
    # @param filename  Path to the file containing the meshes
    #
    # @return A list of Mesh2D
    #         Warning : The process to make them 2D is very naive:
    #                      the Z component is cut.
    #                   So be careful when you create the meshes to test
    #                      (no depth, triangular) and when you export (aligned with the
    #                      plane xy)
    
    options = pyassimp.postprocess.aiProcess_JoinIdenticalVertices \
              | pyassimp.postprocess.aiProcess_Triangulate
    meshesAssimp = pyassimp.load(filename, options)

    meshes2D = []
    for mA in meshesAssimp.meshes:
        # 3D to 2D
        positions = np.reshape(np.reshape(mA.vertices, (-1, 3))[:, :2], (-1))
        indices = np.reshape(mA.faces, (-1))
        meshes2D.append(Mesh2D(positions, indices))

    pyassimp.release(meshesAssimp)
    return meshes2D
    """
