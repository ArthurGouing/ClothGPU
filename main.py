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

import glfw
import numpy as np

from scenes import *
from viewer import Viewer


# Separates the main to make sure all objects are deleted
# before glfw.terminate is called
def main():
    viewer=Viewer(height=960, width=1280,
                  bgColor = np.array([0.2, 0.3, 0.3]), maxFPS=24)

    # Loading the scene
    
    # Parameters
    os.system("PYOPENCL_COMPILER_OUTPUT=1")
    device = "GPU"
    n = 200 # 00x100 = 10 000 | 300x300 = 100 000 | 700x700 = 490 000  
    # Scene
    clothdynamic(viewer, device, n)  # l'objectif c'est 100 !!!
    
    # Main loop
    viewer.run()



if __name__ == '__main__':

    # Initialization
    glfw.init()

    main()
    
    # End
    glfw.terminate()
