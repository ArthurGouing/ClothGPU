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

from .abstract_dynamic_system import AbstractDynamicSystem

## Dummy dynamic system just to test
class DummyDynamicSystem(AbstractDynamicSystem):

    def __init__(self, mesh):
        ## Constructor
        # @param self
        # @param mesh  
        super().__init__()
        self.mesh = mesh

        # Animations parameters
        self.it = 60.
        self.delta = 1.
        self.period = 120.
        self.colours = np.copy(self.mesh.constColours)
        self.translationVector = np.tile([0.01, 0], self.mesh.nbVertices)

    def step(self):

        self.mesh.colours = (self.it / self.period) * self.colours
        self.mesh.positions += self.delta * self.translationVector

        self.it += self.delta
        if (self.it <= 0) or (self.it >= self.period):
            self.delta *= -1.
