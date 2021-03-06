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
        self.A = np.array([[0.8566, -0.3086, 0, 0],
                           [0.0117, 0.7493, 0, 0],
                           [0.0186, 0.00032, 1, 0.4],
                           [0.000125, 0.0174, 0, 1]])
        self.B = np.array([[1.3968], [1.5571], [0.0175], [0.0163]])
        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.R = np.array([[1]])


    def f_xu(self, states, actions):
        # for left turning task the delta_y is y and the delta_phi is phi
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        v_y, r, delta_y, delta_phi, pos_x = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
        steer = actions
        next_state = [self.A[0][0] * v_y + self.A[0][1] * r + self.A[0][2] * delta_y + self.A[0][3] * delta_phi +
                      self.B[0][0] * steer,
                      self.A[1][0] * v_y + self.A[1][1] * r + self.A[1][2] * delta_y + self.A[1][3] * delta_phi +
                      self.B[1][0] * steer,
                      self.A[2][0] * v_y + self.A[2][1] * r + self.A[2][2] * delta_y + self.A[2][3] * delta_phi +
                      self.B[2][0] * steer,
                      self.A[3][0] * v_y + self.A[3][1] * r + self.A[3][2] * delta_y + self.A[3][3] * delta_phi +
                      self.B[3][0] * steer]
        state_new = tf.stack(next_state, 1)
        v_ys, rs, delta_ys, delta_phis = state_new[:, 0], state_new[:, 1], state_new[:, 2], state_new[:, 3]
        pos_xs = pos_x + self.expected_vs * 0.1
        next_states = tf.stack([v_ys, rs, delta_ys, delta_phis, pos_xs], axis=1)
        return next_states


    def prediction(self, x_1, u_1):
        x_next = self.f_xu(x_1, u_1)
        return x_next

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        with tf.name_scope('compute_reward') as scope:
            v_ys, rs, delta_ys, delta_phis = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
            steers = actions
            punish_steer = tf.square(steers)
            devi_y = tf.square(delta_ys)  # distence between current point and clostest point
            devi_phi = tf.square(delta_phis)
            punish_yaw_rate = tf.square(rs)
            rewards = 2 * devi_y + 1 * devi_phi + 0.2 * punish_yaw_rate + 0.5 * punish_steer
        return rewards


class PathTrackingModel(object):  # all tensors
    def __init__(self, num_future_data=0, **kwargs):
        self.vehicle_dynamics = VehicleDynamics()
        self.obses = None
        self.actions = None
        self.num_future_data = num_future_data
        self.expected_vs = 20.
        # self.path = ReferencePath()

    def reset(self, obses, ): #obses???v_y, r, delta_y, delta_phi, x
        self.obses = tf.convert_to_tensor(obses)
        self.actions = None

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        steer_norm = actions[:, 0]
        self.actions = steer_norm * 1.2 * np.pi / 9
        rewards = self.vehicle_dynamics.compute_rewards(self.obses, self.actions)
        self.obses = self.vehicle_dynamics.prediction(self.obses, self.actions)
        v_ys, rs, delta_ys, delta_phis, pos_x = self.obses[:, 0], self.obses[:, 1], self.obses[:, 2], \
                                                   self.obses[:, 3], self.obses[:, 4]
        delta_phis = tf.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
        delta_phis = tf.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
        obses_scale = tf.stack([v_ys, rs, delta_ys, delta_phis], axis=1)
        self.obses = tf.stack([v_ys, rs, delta_ys, delta_phis, pos_x], axis=1)
        return obses_scale, self.obses, rewards

