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

from graphics import *
from dynamics import *
from geom import *


def clothdynamic(viewer, device, n):
    """
    @brief Demo of realtime cloth simulation, to be tested on GPU
    """
    n_points = n*n
    print(f" Run cloth simulation for {n_points} points")

    # Build positions
    positions_builder = list()
    z = 2.1
    for i in range(n):
       x = -5 + float(i)/float(n-1) * 10 
       for j in range(n):
          y = -5 + float(j)/float(n-1) * 10 
          positions_builder.append(x)
          positions_builder.append(y)
          positions_builder.append(z)
    positions = np.array(positions_builder, dtype=np.float64)

    # build colors
    colours_builder = list()
    for i in range(n):
        for j in range(n):
            r = i/(n-1)/2 + (n-j)/n/2
            # r = i/(n-1)/2 + j/(n-1)
            g = j/(n-1)
            b = i/(n-1)
            colours_builder.append(r)
            colours_builder.append(g)
            colours_builder.append(b)
    colours = np.array(colours_builder, dtype=np.float64)

    # Build indices
    indices_builder = list()
    for j in range(n-1):
        for i in range(n-1):
            # Triangle 1
            indices_builder.append(i   + n*j)
            indices_builder.append(i+1 + n*j)
            indices_builder.append(i   + n*(j+1))
            # Triangle 2
            indices_builder.append(i   + n*(j+1))
            indices_builder.append(i+1 + n*j)
            indices_builder.append(i+1 + n*(j+1))
    indices = np.array(indices_builder, dtype=np.float64)

    clothMesh = Mesh3D(positions, indices, colours)
    clothRender = Mesh3DRenderable(clothMesh)

    # Sphere centrée en (0,0,0), de rayon 2
    sphereMesh = Sphere3D(radius=2)
    sphereRender = Sphere3DRenderable(sphereMesh)

    viewer.addRenderable(clothRender)
    viewer.addRenderable(sphereRender)

    clothsim = ClothDynamicSystem(clothMesh)
    # And add it to the viewer
    # Each frame will perform a call to the 'step' method of the viewer
    if device=="CPU":
        clothsim = ClothDynamicSystem(clothMesh)
    elif device=="GPU":
        clothsim = ClothDynamicSystem_GPU(clothMesh)
    else:
        print("Error: the device '{device}' is not supported")
        exit(1)
    viewer.addDynamicSystem(clothsim)


def indexedTest(viewer):
    """
    @brief Demonstration for a basic static rendering
           Renders a simple square 
    """

    # Indexed square
    positions = np.array([0., 0.,   # x0, y0
                          1., 0.,   # x1, y1
                          0., 1.,   # x2, y2
                          1., 1.],  # x3, y3
                         np.float64)
    colours = np.array([1., 0., 0.,  # (r, g, b) for vertex 0
                        0., 0., 1.,  # (r, g, b) for vertex 1
                        0., 1., 0.,  # ...
                        1., 1., 1.]) # ...
    indices = np.array([0, 1, 2,   # First triangle composed by vertices 0, 1 and 2
                        1, 2, 3])  # Second triangle composed by vertices 1, 2 and 3

    # Create the object
    squareMesh = Mesh2D(positions, indices, colours)
    # Create the correspondung GPU object
    squareMeshRenderable = Mesh2DRenderable(squareMesh)
    # Add it to the list of objects to render
    viewer.addRenderable(squareMeshRenderable)

def dynamicTest(viewer):
    """
    @brief Demonstration for a basic dynamic rendering
           Renders a simple square, moved by a dummy dynamic system
    """

    # Indexed square
    positions = np.array([0., 0.,   # x0, y0
                          1., 0.,   # x1, y1
                          0., 1.,   # x2, y2
                          1., 1.],  # x3, y3
                         np.float64)
    colours = np.array([1., 0., 0.,  # (r, g, b) for vertex 0
                        0., 0., 1.,  # (r, g, b) for vertex 1
                        0., 1., 0.,  # ...
                        1., 1., 1.]) # ...
    indices = np.array([0, 1, 2,   # First triangle composed by vertices 0, 1 and 2
                        1, 2, 3])  # Second triangle composed by vertices 1, 2 and 3

    # Create the object
    squareMesh = Mesh2D(positions, indices, colours)
    # Create the correspondung GPU object
    squareMeshRenderable = Mesh2DRenderable(squareMesh)
    # Add it to the list of objects to render
    viewer.addRenderable(squareMeshRenderable)

    # Create a dynamic system
    dyn = DummyDynamicSystem(squareMesh)
    # And add it to the viewer
    # Each frame will perform a call to the 'step' method of the viewer
    viewer.addDynamicSystem(dyn)
    


def rodTest(viewer):

    """
    @brief Demonstration for a rendering of a rod object
           Specific case, as a rod is essentialy a line, we
           need to generate a mesh over it to git it a thickness
           + demonstration of the scaling matrix for the rendering
    """
    positions = np.array([-1., 1.,
                          -1., 0.,
                          -0.5, -0.25],
                         np.float64)
    colours = np.array([1., 0., 0.,
                        0., 1., 0.,
                        0., 0., 1.])

    rod = Rod2D(positions, colours)

    rodRenderable = Rod2DRenderable(rod, thickness = 0.005)
    viewer.addRenderable(rodRenderable)
    
    positionsScaled = np.array([0., 1.,
                                0., 0.,
                                0.5, -0.25],
                               np.float64)
    rodScaled = Rod2D(positionsScaled, colours)

    rodRenderableScaled = Rod2DRenderable(rodScaled, thickness = 0.005)
    rodRenderableScaled.modelMatrix[0, 0] = 2.   # scale in X
    rodRenderableScaled.modelMatrix[1, 1] = 0.75 # scale in Y
    viewer.addRenderable(rodRenderableScaled)
