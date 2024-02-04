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

## Class defining a generic 3D mesh
class Mesh3D(object):
    def __init__(self, positions, indices, colours = None):
        ## Constructor
        # @param positions  1-D Numpy array concatenating
        #                   the 3D positions of the mesh
        # @param indices    1-D Numpy array for the triangles indices (triplets)
        # @param colours    1-D Numpy array concatenating the
        #                   vertices colours (r, g, b)

        self.nbVertices = int(positions.size / 3)
        self.positions = np.array(positions, np.float64)
        self.vertices = self.positions.reshape(-1,3)
        self.velocity = np.zeros(self.vertices.shape)

        self.colours = None
        if (colours is None):
            self.colours = 0.5 * np.ones(3 * self.nbVertices, dtype=np.float32)
        else:
            self.colours = np.array(colours, np.float32)
            if (colours.size != (3 * self.nbVertices)):
                raise Exception("Wrong buffer size")

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

        neigh_size = max([len(neigh) for neigh in self.neighbours])
        print(neigh_size)
        self.l0 = list()
        for vi, neigh in zip(self.vertices, self.neighbours):
            l0i = neigh_size * [0]
            for i, j in enumerate(neigh):
                vj = self.vertices[j]
                l0 = np.linalg.norm(vj-vi)
                l0i[i] = l0
            self.l0.append(l0i)


        # Fields to optimise out unnecessary redraw
        self.positionsUpdated = True
        self.coloursUpdated = True
    
    def __setattr__(self, name, value):
        ## Attribute setter
        ## Overloaded to mark positions or colours as updated when changed
        if (name == "positions"):
            self.positionsUpdated = True
        elif (name == "colours"):
            self.coloursUpdated = True
        
        object.__setattr__(self, name, value)
            