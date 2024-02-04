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
from math import sqrt

from .abstract_dynamic_system import AbstractDynamicSystem

## Dummy dynamic system just to test
class ClothDynamicSystem(AbstractDynamicSystem):

    def __init__(self, mesh):
        ## Constructor
        # @param self
        # @param mesh  
        super().__init__()
        self.mesh = mesh


        # Animations parameters
        self.substep = 100
        self.delta = 1./(self.substep * 24.)
        self.E = 1e7
        self.l0 = 10/(sqrt(mesh.nbVertices)-1)
        self.k = 1000
        self.radius = 2.
        self.center = np.array([0,0,0], np.float64)
        self.g = -9.81
        self.translationVector = np.tile([0.01, 0, 0], self.mesh.nbVertices)

        print(f"Parameters: ")
        print("  E: ", self.E)
        print("  l0:", self.l0)
        print("  k: ", self.k)

    def step(self):
        # detect Collision
        for s in range(self.substep):
            collider = self.detect_collision()
            force = np.zeros((self.mesh.nbVertices,3))
            self.compute_volumic_force(force)
            self.compute_surfacic_force(force, collider)
            self.compute_internal_force(force)
            self.solve(force)
            self.mesh.positions = self.mesh.vertices.reshape(-1)

    def detect_collision(self):
        collider = list()
        for i, vert in enumerate(self.mesh.vertices):
            if np.linalg.norm(vert-self.center) < self.radius:
                c = {"id":i, 
                     "P": vert, 
                     "dir":(vert-self.center)/np.linalg.norm(vert), 
                     "dist":abs(np.linalg.norm(vert-self.center)-self.radius)}
                collider.append(c)
        return collider

    def compute_volumic_force(self, force):
        for i in range(self.mesh.vertices.shape[0]):
            force[i] += np.array([0, 0, self.g], np.float64)

    def compute_surfacic_force(self, forces, collider):
        for c in collider:
            i = c["id"]
            forces[i] += self.E * self.radius**(1/2) * c["dist"]**(3/2) * c["dir"]

    def compute_internal_force(self, force):
        for i, vi in enumerate(self.mesh.vertices):

            for j, l0 in zip(self.mesh.neighbours[i], self.mesh.l0[i]):
                vj = self.mesh.vertices[j]
                l = np.linalg.norm(vj-vi)
                dir = (vj-vi)/l
                force[i] +=  self.k * (l-l0) * dir
                force[j] += -self.k * (l-l0) * dir

    def solve(self, force):
        self.mesh.velocity += self.delta * force
        self.mesh.vertices += self.delta * self.mesh.velocity

