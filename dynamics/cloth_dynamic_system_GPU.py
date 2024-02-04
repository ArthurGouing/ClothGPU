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
from time import perf_counter

import pyopencl as cl

from .abstract_dynamic_system import AbstractDynamicSystem

## Dummy dynamic system just to test
class ClothDynamicSystem_GPU(AbstractDynamicSystem):

    def __init__(self, mesh):
        ## Constructor
        # @param self
        # @param mesh  
        super().__init__()
        self.mesh = mesh

        # Init OpenCL
        platforms = cl.get_platforms()
        devices = []
        for p in platforms:
            devices += p.get_devices()
        print("Devices: ",devices)
        print(platforms)
        platform = platforms[0]
        ctx = cl.Context(
            dev_type=cl.device_type.GPU,
            properties=[(cl.context_properties.PLATFORM, platform)])
        # Print GPU specs
        print("{0:<20}: {1}".format("OpenCL", platform.version))
        print("{0:<20}: {1}".format("Profile", platform.profile))
        # print("Extension:", str(platform.extensions.strip().Split(' ')))
        dev = platform.get_devices(cl.device_type.ALL)[0]
        print("{0:<20}: {1}".format("Device", dev.name))
        print('{} ({})'.format(dev.name, dev.vendor))
        flags = [('Version', dev.version),
                 ('Type', cl.device_type.to_string(dev.type)),
                 # ('Extensions', str(dev.extensions.strip().split(' '))),
                 ('Memory (global)', str(dev.global_mem_size/1e9) + " Gbyte"),
                 ('Memory (local)', str(dev.local_mem_size/1e3) + " Kbyte"),
                 ('Address bits', str(dev.address_bits)),
                 ('Max work item dims', str(dev.max_work_item_dimensions)),
                 ('Max work group size', str(dev.max_work_group_size)),
                 ('Max compute units', str(dev.max_compute_units)),
                ]
        [print('{0:<20}: {1:<10}'.format(name, flag)) for name, flag in flags]

        self.gpu_queue = cl.CommandQueue(ctx)
        program = cl.Program(ctx, open('dynamics/cloth_compute.cl').read()).build(options='')

        # Create buffers :
        mf = cl.mem_flags
        # replace by get from OpenGL buffer
        self.positions_h = np.float32(self.mesh.positions)
        self.positions_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.positions_h)
        self.velocity_h = np.float32(self.mesh.velocity.reshape(-1))
        self.velocity_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.velocity_h)
        self.l0_h = np.float32(self.mesh.l0).reshape(-1)
        self.l0_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.l0_h)
        self.neighbour_h = list()
        i=0
        max_neighour = max([len(neigh) for neigh in self.mesh.neighbours])
        for neigh in self.mesh.neighbours:
            i_neighbours = max_neighour * [-1]
            i_neighbours[:len(neigh)] = neigh
            self.neighbour_h += i_neighbours
            if i==101:
                print(neigh)
                print (self.neighbour_h)
            i += 1
        self.neighbour_h = np.int32(self.neighbour_h)
        print("neighbours size: ",self.neighbour_h.size)
        print("neighbours shape: ",self.neighbour_h.shape)
        self.neighbour_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.neighbour_h)

        self.kernel = program.step  # Use this Kernel object for repeated calls
        self.global_work_group_size = (self.mesh.nbVertices,1,1)
        self.local_work_group_size = None
        # self.local_size = (1024, 1, 1) if self.positions_h.size > 1024 else (self.positions_h.size, 1, 1)
        # positions
        # velocity
        # neighboor
        # forces

        # Animations parameters
        self.substep = 250
        self.delta = 1./(self.substep * 24.)
        self.E = 1e5 # Contact (young modulus of sphere)
        self.l0 = 10/(sqrt(mesh.nbVertices)-1)
        self.k = 8e4 #Internal (rididity of inner spring)
        self.radius = 2.
        self.center = np.array([0,0,0], np.float64)
        self.g = -9.81

        print(f"Parameters: ")
        print("  dt:", self.delta)
        print("  E: ", self.E)
        print("  l0:", self.l0)
        print("  k: ", self.k)
        print("  r: ", self.radius)
        print("  shape:", self.positions_h.shape, "type: ", type(self.positions_h.shape))

    def step(self):
        t_start = perf_counter()
        for s in range(self.substep):
            self.kernel.set_args(self.positions_d, self.velocity_d, self.l0_d, self.neighbour_d,
                                  np.float32(self.delta), np.float32(self.E), np.float32(self.k), np.float32(self.radius))
            event = cl.enqueue_nd_range_kernel(self.gpu_queue, self.kernel, self.global_work_group_size, self.local_work_group_size)
            #event.wait()
        cl.enqueue_copy(self.gpu_queue, self.positions_h, self.positions_d)
        t_end = perf_counter()
        print(f"GPU Computation time: {1000 * (t_end-t_start):.3f} ms")

        self.mesh.positions = self.positions_h

        # for s in range(self.substep):
        #     collider = self.detect_collision()
        #     force = np.zeros((self.mesh.nbVertices,3))
        #     self.compute_volumic_force(force)
        #     self.compute_surfacic_force(force, collider)
        #     self.compute_internal_force(force)
        #     self.solve(force)
        #     self.mesh.positions = self.mesh.vertices.reshape(-1)

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
