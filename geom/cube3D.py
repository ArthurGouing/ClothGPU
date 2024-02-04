#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# This file is part of SimulationTeachingElan, a python code used for teaching at Elan Inria.
#
# Copyright 2022 Thibaut Metivet <thibaut.metivet@inria.fr> (Elan / Inria - Université Grenoble Alpes)
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

from .rigidBody3D import RigidBody3D

## Class defining a 3D cube
class Cube3D(RigidBody3D):
    def __init__(self, center=[0,0,0], rotation=np.eye(3,3), lengths=[1,1,1] ):
        ## Constructor
        # @param center     1-D Numpy array with 3D position of the center
        # @param rotation   1-D Numpy array for the triangles indices (triplets)

        super().__init__(center, rotation)
        self.lengths = np.array(lengths, np.float64)
        self.nbVertices = 8
        refVertices = np.array(
            [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
             [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]]
             , np.float64)
        self.refVertices = refVertices * 0.5 * self.lengths

        self.vertices = np.array( (self.nbVertices,3), np.float64 )
        self.verticesUpToDate = False
        self.updateVertices()

        # # Build neighbours list
        # self.neighbours = [ set() for i in range(self.nbVertices)]
        # for i in range(int(self.indices.size / 3)):
        #     t = self.indices[3 * i : 3 * i + 3]
        #     for j in range(3):
        #         self.neighbours[t[j]].add(t[(j + 1) % 3])
        #         self.neighbours[t[j]].add(t[(j + 2) % 3])
        # for i in range(self.nbVertices):
        #     self.neighbours[i] = list(self.neighbours[i])
    
    def __getattribute__(self, name):
        if (name == "vertices" and not self.verticesUpToDate):
            self.updateVertices()
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        ## Attribute setter
        ## Overloaded to mark positions or colours as updated when changed
        if (name == "center" or name == "rotation"):
            self.verticesUpToDate = False
        object.__setattr__(self, name, value)

    def updateVertices(self):
        self.vertices = self.center + np.dot( self.refVertices, self.rotation.transpose() )
        self.verticesUpToDate = True
            