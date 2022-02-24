#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: path_tracking_env.py
# =====================================

from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class VehicleDynamics(object):
    def __init__(self):
        self.expected_vs = 20.
        # self.path = ReferencePath()
        self.A = np.array([[0.4411, -0.6398, 0, 0],
                           [0.0242, 0.2188, 0, 0],
                           [0.0703, 0.0171, 1, 2],
                           [0.0018, 0.0523, 0, 1]])
        self.B = np.array([[2.0350], [4.8124], [0.4046], [0.2952]])

    def f_xu(self, states, actions):
        # for left turning task the delta_y is y and the delta_phi is phi
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        v_y, r, delta_y, delta_phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        steer = actions
        next_state = [self.A[0][0] * v_y + self.A[0][1] * r + self.A[0][2] * delta_y + self.A[0][3] * delta_phi +
                      self.B[0][0] * steer,
                      self.A[1][0] * v_y + self.A[1][1] * r + self.A[1][2] * delta_y + self.A[1][3] * delta_phi +
                      self.B[1][0] * steer,
                      self.A[2][0] * v_y + self.A[2][1] * r + self.A[2][2] * delta_y + self.A[2][3] * delta_phi +
                      self.B[2][0] * steer,
                      self.A[3][0] * v_y + self.A[3][1] * r + self.A[3][2] * delta_y + self.A[3][3] * delta_phi +
                      self.B[3][0] * steer]
        # print('states = ', states)
        # print('actions = ', actions)
        # print('next_state_0 = ', self.A[0][0] * v_y + self.A[0][1] * r + self.A[0][2] * delta_y + self.A[0][3] * delta_phi +
        #               self.B[0][0] * steer)
        # print('next_state_1 = ', self.A[1][0] * v_y + self.A[1][1] * r + self.A[1][2] * delta_y + self.A[1][3] * delta_phi +
        #               self.B[1][0] * steer)
        # print('next_state_2 = ', self.A[2][0] * v_y + self.A[2][1] * r + self.A[2][2] * delta_y + self.A[2][3] * delta_phi +
        #               self.B[2][0] * steer)
        # print('next_state_3 = ', self.A[3][0] * v_y + self.A[3][1] * r + self.A[3][2] * delta_y + self.A[3][3] * delta_phi +
        #               self.B[3][0] * steer)
        # print('next_state = ', tf.stack(next_state, 1))
        return tf.stack(next_state, 1)


    def prediction(self, x_1, u_1):
        x_next = self.f_xu(x_1, u_1)
        return x_next

    # def simulation(self, states, actions):
    #     # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, xs
    #     # veh_full_state: v_xs, v_ys, rs, ys, phis, xs
    #     # others: alpha_f, alpha_r, r, alpha_f_bounds, alpha_r_bounds, r_bounds
    #     #for i in range(simu_times):
    #     states = tf.convert_to_tensor(states.copy(), dtype=tf.float32)
    #     states = self.prediction(states, actions)
    #     states = states.numpy()
    #     states[:, -1][states[:, -1] > self.path.period] -= self.path.period
    #     states[:, -1][states[:, -1] <= 0] += self.path.period
    #     states[:, 3][states[:, 3] > np.pi] -= 2 * np.pi
    #     states[:, 3][states[:, 3] <= -np.pi] += 2 * np.pi
    #
    #     return states

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        with tf.name_scope('compute_reward') as scope:
            v_ys, rs, delta_ys, delta_phis = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
            # print()
            steers = actions
            punish_steer = tf.square(steers)
            devi_y = tf.square(delta_ys)  # distence between current point and clostest point
            devi_phi = tf.square(delta_phis)
            punish_yaw_rate = tf.square(rs)
            rewards = 4 * devi_y + 0.1 * devi_phi + 0.2 * punish_yaw_rate + 0.5 * punish_steer
        return rewards


# class ReferencePath(object):
#     def __init__(self):
#         self.period = 1200.
#         self.path_x = self.compute_x_point()
#         self.path_y = self.compute_path_y(self.path_x)
#         self.path_phi = self.compute_path_phi(self.path_x)
#
#
#     def compute_path_y(self, x):
#         y = np.zeros_like(x, dtype=np.float32)
#         return y
#
#     def compute_path_phi(self, x):
#         deriv = np.zeros_like(x, dtype=np.float32)
#         return deriv
#
#     # def compute_y(self, x, delta_y):
#     #     y_ref = self.compute_path_y(x)
#     #     return delta_y + y_ref
#     #
#     # def compute_delta_y(self, x, y):
#     #     y_ref = self.compute_path_y(x)
#     #     return y - y_ref
#
#     # def compute_phi(self, x, delta_phi):
#     #     phi_ref = self.compute_path_phi(x)
#     #     phi = delta_phi + phi_ref
#     #     phi[phi > np.pi] -= 2 * np.pi
#     #     phi[phi <= -np.pi] += 2 * np.pi
#     #     return phi
#
#     def compute_x_point(self):
#         x = np.linspace(0., 1000., 10000).astype(np.float32)
#         return x


class PathTrackingModel(object):  # all tensors
    def __init__(self, num_future_data=0, **kwargs):
        self.vehicle_dynamics = VehicleDynamics()
        self.obses = None
        self.actions = None
        self.num_future_data = num_future_data
        self.expected_vs = 20.
        # self.path = ReferencePath()

    def reset(self, obses, ): #obses为v_y, r, delta_y, delta_phi, x
        self.obses = tf.convert_to_tensor(obses)
        self.actions = None

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        steer_norm = actions[:, 0]
        self.actions = steer_norm * 1.2 * np.pi / 9
        # print('states = ', self.obses)
        rewards = self.vehicle_dynamics.compute_rewards(self.obses, self.actions)
        self.obses = self.vehicle_dynamics.prediction(self.obses, self.actions)
        # print('self.obses = ', self.obses)
        # self.obses[:, -1] = self.obses[:, -1] + self.expected_vs * 0.1
        # delta_phis = self.obses[:, 3]
        v_ys, rs, delta_ys, delta_phis = self.obses[:, 0], self.obses[:, 1], self.obses[:, 2], \
                                                   self.obses[:, 3]
        delta_phis = tf.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
        delta_phis = tf.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
        self.obses = tf.stack([v_ys, rs, delta_ys, delta_phis], axis=1)
        return self.obses, rewards

