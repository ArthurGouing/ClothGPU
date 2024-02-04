#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# This file is part of SimulationTeachingElan, a python code used for teaching at Elan Inria.
#
# Copyright 2020 Mickael Ly <mickael.ly@inria.fr> (Elan / Inria - Université Grenoble Alpes)
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

import OpenGL.GL as GL
import glfw
import numpy as np
import math

## Simple 2D Camera
class Camera:
    def __init__(self, window, distance = 2.):
        ## Constructor
        # Link the mouse callback to the camera
        # @param self
        # @param window

        # Init Camera
        self.pos = np.array([0., 0.])
        self.distanceMin = 1.e-3
        self.distance = max(self.distanceMin, distance)
        # Callbacks
        self.mousePos = np.array([0., 0.])
        glfw.set_cursor_pos_callback(window, self.onMouseMove)
        glfw.set_scroll_callback(window, self.onMouseScroll)

    def onMouseMove(self, window, newXPos, newYPos):
        ## Mouse drag callback
        # Translate with a right click
        # @param self
        # @param window
        # @param newXPos
        # @param newYPos
        oldPos = self.mousePos
        self.mousePos = np.array([newXPos, glfw.get_window_size(window)[1] - newYPos])
        #if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
        #    self.drag(oldPos, self.mousePos, glfw.get_window_size(window))
        #    return
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT):
            self.translate(oldPos, self.mousePos)

    def onMouseScroll(self, window, deltaX, deltaY):
        ## Mouse scroll callback
        # Zoom / unzoom
        # @param self
        # @param win
        # @param deltax
        # @param deltay
        self.zoom(deltaY, glfw.get_window_size(window)[1])

    def zoom(self, delta, size):
        ## Zoom method
        # @param self
        # @param delta
        # @param size
        zoomSpeed = 50.
        self.distance = max(self.distanceMin,
                            self.distance * (1 - zoomSpeed*delta/size))

    def translate(self, oldPos, newPos):
        ## Translate method
        # @param self
        # @param oldPos
        # @param newPos
        translateSpeed = 2.e-3
        self.pos += translateSpeed * (newPos - oldPos) * self.distance

    def viewMatrix(self):
        ## Compute the view matrix
        # @param self
        # @return The view matrix
        vM = np.identity(4, np.float32)
        vM[0:2, 3] = self.pos
        vM[2, 3] = -self.distance
        return vM

    def projectionMatrix(self, windowSize):
        ## Compute the projection matrix
        # http://www.songho.ca/opengl/gl_projectionmatrix.html
        # @param self
        # @param windowSize
        # @return The projection matirx

        # Clipping
        zNear = 0.01 * self.distance
        zFar = 100 * self.distance

        aspectRatio = windowSize[0] / windowSize[1]

        """
        # Perspective matrix
        viewAngle = 35.
        sY = 1. / (math.tan(math.radians(viewAngle) / 2.))
        sX = sY / aspectRatio
        """
        # Orthographic
        sY = 1.
        sX = 1. / aspectRatio
        
        zE = (zNear + zFar) / (zNear - zFar)
        zN = 2. * zFar * zNear / (zNear - zFar)
        
        return np.array([[sX, 0,  0,  0],
                         [0,  sY, 0,  0],
                         [0,  0, zE, zN],
                         [0,  0, -1,  0]], dtype="float")

class OrbitCamera3D:
    def __init__(self, window, center=[0,0,0], distance = 2., 
        azimuth=-0.5*np.pi, polar=0.5*np.pi):
        ## Constructor
        # Link the mouse callback to the camera
        # @param self
        # @param window

        # Init Camera
        self.center = np.array(center, np.float32)
        self.distanceMin = 1.e-3
        self.distance = max(self.distanceMin, distance)
        self.azimuthAngle = azimuth
        self.polarAngle = polar
        self.localPos = np.array([0., 0.], np.float32) # position in (U,V) plane
        # Callbacks
        self.mousePos = np.array([0., 0.])
        glfw.set_cursor_pos_callback(window, self.onMouseMove)
        glfw.set_scroll_callback(window, self.onMouseScroll)
        # Mouse control speeds
        self.orbitSpeed = 0.01

    def onMouseMove(self, window, newXPos, newYPos):
        ## Mouse drag callback
        # Translate with a right click
        # @param self
        # @param window
        # @param newXPos
        # @param newYPos
        oldPos = self.mousePos
        self.mousePos = np.array([newXPos, glfw.get_window_size(window)[1] - newYPos])
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT):
            self.translate(oldPos, self.mousePos)
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
            dPos = self.mousePos - oldPos
            self.orbit(dPos[0]*self.orbitSpeed, -dPos[1]*self.orbitSpeed)

    def onMouseScroll(self, window, deltaX, deltaY):
        ## Mouse scroll callback
        # Zoom / unzoom
        # @param self
        # @param win
        # @param deltax
        # @param deltay
        self.zoom(deltaY, glfw.get_window_size(window)[1])

    def zoom(self, delta, size):
        ## Zoom method
        # @param self
        # @param delta
        # @param size
        zoomSpeed = 50.
        self.distance = max(self.distanceMin,
                            self.distance * (1 - zoomSpeed*delta/size))

    def translate(self, oldPos, newPos):
        ## Translate method
        # @param self
        # @param oldPos
        # @param newPos
        translateSpeed = 2.e-3
        self.localPos += translateSpeed * (newPos - oldPos) * self.distance

    def orbit(self, angleU, angleV):
        ## Orbit method
        # @param self
        # @param angleU (azimutal increment angle)
        # @param angleV (polar increment angle)
        self.azimuthAngle += angleU
        self.polarAngle += angleV

    def lookAtX(self):
        ## Align camera to X
        self.azimuthAngle = 0.
        self.polarAngle = 0.
        self.localPos.fill(0)
    def lookAtY(self):
        ## Align camera to y
        self.azimuthAngle = 0.5*np.pi
        self.polarAngle = 0.
        self.localPos.fill(0)
    def lookAtZ(self):
        ## Align camera to y
        self.azimuthAngle = -0.5*np.pi
        self.polarAngle = 0.5*np.pi
        self.localPos.fill(0)
    def lookAtMinus(self):
        self.azimuthAngle += np.pi
        self.polarAngle += np.pi

    def viewMatrix(self):
        ## Compute the view matrix
        # @param self
        # @return The view matrix
        vM = np.empty( (4, 4), np.float32)
        cosA = math.cos(self.azimuthAngle)
        sinA = math.sin(self.azimuthAngle)
        cosP = math.cos(self.polarAngle)
        sinP = math.sin(self.polarAngle)
        U = np.array( [-sinA, cosA, 0.], np.float32)
        V = np.array( [-cosA*sinP, -sinP*sinA, cosP], np.float32)
        N = np.array( [cosP*cosA, cosP*sinA, sinP], np.float32)
        position = self.center + self.distance * N \
            - self.localPos.dot(np.array([U.transpose(), V.transpose()]))
        vM[0,0:3] = U
        vM[1,0:3] = V
        vM[2,0:3] = N
        vM[0:3, 3] = - vM[0:3,0:3].dot(position)
        vM[3,0:3] = 0
        vM[3,3] = 1.0
        return vM

    def projectionMatrix(self, windowSize):
        ## Compute the projection matrix
        # @param self
        # @param windowSize
        # @return The projection matirx

        # Clipping
        zNear = 0.01 * self.distance
        zFar = 100 * self.distance

        aspectRatio = windowSize[0] / windowSize[1]

        """
        # Perspective matrix
        viewAngle = 35.
        sY = 1. / (math.tan(math.radians(viewAngle) / 2.))
        sX = sY / aspectRatio
        """
        # Orthographic
        sY = 1.
        sX = 1. / aspectRatio
        
        zE = (zNear + zFar) / (zNear - zFar)
        zN = 2. * zFar * zNear / (zNear - zFar)
        
        return np.array([[sX, 0,  0,  0],
                         [0,  sY, 0,  0],
                         [0,  0, zE, zN],
                         [0,  0, -1,  0]], dtype="float")