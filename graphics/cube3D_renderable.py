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

## Class rendering a 3D cube
class Cube3DRenderable(AbstractRenderable):
    def __init__(self, cube):
        ## Constructor
        # Initialize the GPU buffers required for a cube
        # @param self
        # @param cube

        super().__init__()
        
        # Cube object
        self.cube = cube

        # Triangle indices
        self.indices = np.array(
            [ 0,1,2, 0,2,3,  0,3,4, 0,4,5,  0,5,6, 0,6,1,
              1,6,7, 1,7,2,  7,4,3, 7,3,2,  4,7,6, 4,6,5 ],
              np.uint32
        )
        self.doUpdatePositionBuffer = True

        # Colour
        self.colours = np.array(
            [1., 0., 0.,  # (r, g, b) for vertex 0
             1., 1., 0.,  # (r, g, b) for vertex 1
             0., 1., 0.,
             0., 0., 1.,
             0., 1., 1.,
             1., 0., 1.,
             0., 0., 0.,
             1., 1., 1.] )
        self.doUpdateColourBuffer = True

        # Weak protection for the buffers update
        self.vaoBound = False
        
        # Create the VAO
        self.glId = GL.glGenVertexArrays(1)
        self.bindVAO()

        # VBOs
        ## Ugly -- assumes the locations
        
        # Positions
        # (doubles needed for the simulations)
        # Data storing
        self.locations["positions"] = 0 # <--
        self.buffers["positions"] = GL.glGenBuffers(1)
        # Send data
        self.updatePositionsBuffer()
        # Drawing instructions
        GL.glVertexAttribPointer(self.locations["positions"], 3,
                                 GL.GL_DOUBLE, False, 0, None)

        # Data
        self.locations["colours"] = 1 # <--
        self.buffers["colours"] = GL.glGenBuffers(1)
        # Send data
        self.updateColoursBuffer()
        # Drawing instructions
        GL.glVertexAttribPointer(self.locations["colours"], 3,
                                 GL.GL_FLOAT, False, 0, None)

        # Indexed drawing
        # Data
        self.buffers["indices"] = GL.glGenBuffers(1)
        # Drawing instructions
        indicesId = self.buffers["indices"]
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, indicesId)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indices, GL.GL_STATIC_DRAW)
        self.drawCommand = GL.glDrawElements
        self.drawArguments = (self.indices.size, GL.GL_UNSIGNED_INT, None)

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
            self.doUpdateColourBuffer = True
            return object.__setattr__(self, "colours", value)
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

    def updatePositionsBuffer(self):
        ## Update the GPU positions buffer
        ## VAO of this object must be bind before calling
        # @param self
        if (not self.doUpdatePositionBuffer and self.cube.verticesUpToDate) \
            or (not self.vaoBound):
            return
        self.doUpdatePositionBuffer = False
        positionLocation = self.locations["positions"]
        positions = self.cube.vertices.reshape(-1)
        positionId = self.buffers["positions"]
        GL.glEnableVertexAttribArray(positionLocation)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, positionId)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, positions,
                        GL.GL_STATIC_DRAW)
        
    def updateColoursBuffer(self):
        ## Update the GPU colour buffer
        ## VAO of this object must be bind before calling
        # @param self
        if (not self.doUpdateColourBuffer) or (not self.vaoBound):
            return
        self.doUpdateColourBuffer = False
        colourLocation = self.locations["colours"]
        colours = np.array(self.colours, \
                           np.float32, copy=False)
        colourId = self.buffers["colours"]
        GL.glEnableVertexAttribArray(colourLocation)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, colourId)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colours, GL.GL_STATIC_DRAW)
    


    def draw(self, viewMatrix, projectionMatrix,
             shaderProgram, primitive = GL.GL_TRIANGLES):
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
                 "projectionMatrix"]
        locations = {n: GL.glGetUniformLocation(shaderProgram.glId, n)
                     for n in names}
        GL.glUseProgram(shaderProgram.glId)       

        
        GL.glUniformMatrix4fv(locations["modelMatrix"], 1, True, self.modelMatrix)
        GL.glUniformMatrix4fv(locations["viewMatrix"], 1, True, viewMatrix)
        GL.glUniformMatrix4fv(locations["projectionMatrix"], 1, True, projectionMatrix)

        self.bindVAO()

        self.updatePositionsBuffer()
        self.updateColoursBuffer()
        
        self.drawCommand(primitive, *self.drawArguments)

        self.releaseVAO()
            
            
    def __del__(self):
        super().__del__()
