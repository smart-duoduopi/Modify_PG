#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/1/10
# @Author  : Jiaxin Gao (Tsinghua Univ.)
# =====================================
import numpy as np

class VehicleDynamics(object):
    def __init__(self):
        self.A = None
        self.B = None
        self.Q = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 36, 0],
                      [0, 0, 0, 1]])
        self.R = 1
        self.system_matrix()

    def system_matrix(self):  # states and actions are tensors, [[], [], ...]
       self.A = np.array([[0.4411, -0.6398, 0, 0],
                      [0.0242, 0.2188, 0, 0],
                      [0.0703, 0.0171, 1, 2],
                      [0.0018, 0.0523, 0, 1]])
       self.B = np.array([[0.1163], [0.2750], [0.0231], [0.0169]])


if __name__ == '__main__':
    vehicle = VehicleDynamics()
    print(vehicle.A)


