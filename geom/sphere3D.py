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

## Class defining a 3D sphere
class Sphere3D(RigidBody3D):
    def __init__(self, center=[0,0,0], rotation=np.eye(3,3), radius=1.0 ):
        ## Constructor
        # @param center     1-D Numpy array with 3D position of the center
        # @param rotation   1-D Numpy array for the triangles indices (triplets)
        # @param radius     float: radius of the sphere

        super().__init__(center, rotation)
        self.radius = radius
    