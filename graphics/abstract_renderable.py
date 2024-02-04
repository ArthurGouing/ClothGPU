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
import OpenGL.GL as GL
from abc import ABCMeta, abstractmethod
import numpy as np


## Abstract class defining an object to render
class AbstractRenderable(metaclass=ABCMeta):
    
    def __init__(self):
        ## Constructor
        # Should allocate the buffers
        # @param self

        # Id of the VAO
        self.glId = None
        # Buffers in a dict
        self.buffers = {}
        self.locations = {}

    @abstractmethod
    def draw(self, modelMatrix, viewMatrix, projectionMatrix,
             shaderProgram, primitive = GL.GL_TRIANGLES):
        ## Call to draw
        # @param self
        pass


    def __del__(self):
        ## Desctructor
        # Release the buffers
        # @param self
        if self.glId is not None:
            GL.glDeleteVertexArrays(1, np.array([self.glId]))
            GL.glDeleteBuffers(len(self.buffers),
                               np.array([buf for buf in self.buffers.values()]))
