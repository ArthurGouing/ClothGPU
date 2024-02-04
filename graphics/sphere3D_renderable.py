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

from .abstract_renderable import AbstractRenderable
import OpenGL.GL as GL
import numpy as np

from graphics.shader import Shader

## Class rendering a 3D cube
class Sphere3DRenderable(AbstractRenderable):
    def __init__(self, sphere):
        ## Constructor
        # Initialize the GPU buffers required for a sphere
        # @param self
        # @param sphere

        super().__init__()
        
        # Sphere object
        self.sphere = sphere

        # Impostor cube vertices
        self.vertices = np.array( [ 
            -1., 1., 0.,
            1., 1., 0.,
            -1., -1., 0., 
            1., -1., 0.
            ]
        )

        # Octant colors, depending on Z,Y,X normals
        self.colors = np.array( [
            1., 0., 0.,  # (r, g, b) for (+,+,+)
            1., 1., 0.,  # (r, g, b) for (+,+,-)
            0., 1., 0.,  # (r, g, b) for (+,-,+)
            0., 0., 1.,  # ...
            0., 1., 1.,
            1., 0., 1.,
            0., 0., 0.,
            1., 1., 1. ]
            )

        # Weak protection for the buffers update
        self.vaoBound = False
        
        # Create the VAO
        self.glId = GL.glGenVertexArrays(1)
        self.bindVAO()

        # VBOs
        ## Ugly -- assumes the locations
        
        # Impostor vertices
        # Data buffer
        self.locations["vertices"] = 0 # <--
        self.buffers["vertices"] = GL.glGenBuffers(1)
        # Send data
        GL.glEnableVertexAttribArray(self.locations["vertices"])
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers["vertices"])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices,
                        GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(self.locations["vertices"], 3,
                                 GL.GL_FLOAT, False, 0, None)
        
        # # Drawing instructions
        self.drawCommand = GL.glDrawArrays
        self.drawArguments = (0, self.vertices.size)

        # End of the VAO commands -- unbind everything
        self.releaseVAO()
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        
        # Model (scaling) matrix
        self.modelMatrix = np.identity(4, dtype="float")

    def __setattr__(self, name, value):
        ## Attribute setter
        ## Overloaded to mark colours as updated when changed
        if ((name == "colours") or (name == "colors")):
            return object.__setattr__(self, "colors", value)
        else: 
            object.__setattr__(self, name, value)

    def bindVAO(self):
        ## Bind the VAO of this object 
        ## Can still be hacked through calls to bindVAO of other objects...
        # @param self
        GL.glBindVertexArray(self.glId)
        self.vaoBound = True

    def releaseVAO(self):
        ## Release the VAO of this object
        # @param self
        GL.glBindVertexArray(0)
        self.vaoBound = False

    def draw(self, viewMatrix, projectionMatrix,
             shaderProgram, primitive = GL.GL_TRIANGLE_STRIP):
        ## Drawing function
        # @param self
        # @param projectionMatrix
        # @param viewMatrix
        # @param modelMatrix
        # @param shaderProgram
        # @param primitive
        
        # Send uniforms
        names = ["modelMatrix",
                 "viewMatrix",
                 "projectionMatrix",
                 "colors",
                 "center",
                 "radius",
                 "rotation"]
        locations = {n: GL.glGetUniformLocation(shaderProgram.glId, n)
                     for n in names}
        GL.glUseProgram(shaderProgram.glId)       

        
        GL.glUniformMatrix4fv(locations["modelMatrix"], 1, True, self.modelMatrix)
        GL.glUniformMatrix4fv(locations["viewMatrix"], 1, True, viewMatrix)
        GL.glUniformMatrix4fv(locations["projectionMatrix"], 1, True, projectionMatrix)
        GL.glUniform3fv(locations["colors"], 8, self.colors)
        
        GL.glUniform3fv(locations["center"], 1, self.sphere.center)
        GL.glUniform1f(locations["radius"], self.sphere.radius)
        GL.glUniformMatrix3fv( locations["rotation"], 1, False, self.sphere.rotation )

        self.bindVAO()
 
        self.drawCommand(primitive, *self.drawArguments)

        self.releaseVAO()
            
            
    def __del__(self):
        super().__del__()

