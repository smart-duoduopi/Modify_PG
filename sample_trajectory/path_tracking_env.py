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
        self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.expected_vs = 2.
        self.vehicle_params.update(dict(F_zf=F_zf, F_zr=F_zr))
        self.path = ReferencePath()
        # self.steer = None

    def f_xu(self, states, actions, tau):
        # for left turning task the delta_y is y and the delta_phi is phi
        v_x, v_y, r, delta_y, delta_phi, x = states[:, 0], states[:, 1], states[:, 2], \
                                             states[:, 3], states[:, 4], states[:, 5]
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
        C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
        a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
        b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
        mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
        I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * tf.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (tf.square(a) * C_f + tf.square(b) * C_r) - I_z * v_x),
                      delta_y + tau * (v_x * tf.sin(delta_phi) + v_y * tf.cos(delta_phi)),
                      delta_phi + tau * r,
                      x + tau * (v_x * tf.cos(delta_phi) - v_y * tf.sin(delta_phi)),
                      ]

        return tf.stack(next_state, 1)

    def prediction(self, x_1, u_1, frequency):
        x_next = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next

    def simulation(self, states, actions, base_freq):
        # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, xs
        # veh_full_state: v_xs, v_ys, rs, ys, phis, xs
        # others: alpha_f, alpha_r, r, alpha_f_bounds, alpha_r_bounds, r_bounds
        #for i in range(simu_times):
        states = tf.convert_to_tensor(states.copy(), dtype=tf.float32)
        states = self.prediction(states, actions, base_freq)
        states = states.numpy()
        states[:, 0] = np.clip(states[:, 0], 1, 35)
        # v_xs, v_ys, rs, phis = full_states[:, 0], full_states[:, 1], full_states[:, 2], full_states[:, 4]
        # full_states[:, 4] += rs / base_freq
        # full_states[:, 3] += (v_xs * np.sin(phis) + v_ys * np.cos(phis)) / base_freq
        # full_states[:, -1] += (v_xs * np.cos(phis) - v_ys * np.sin(phis)) / base_freq
        # full_states[:, 0:3] = states[:, 0:3].copy()
        # path_y, path_phi = self.path.compute_path_y(full_states[:, -1]), \
        #                    self.path.compute_path_phi(full_states[:, -1])
        # states[:, 4] = full_states[:, 4] - path_phi
        # states[:, 3] = full_states[:, 3] - path_y
        # full_states[:, 4][full_states[:, 4] > np.pi] -= 2 * np.pi
        # full_states[:, 4][full_states[:, 4] <= -np.pi] += 2 * np.pi
        # full_states[:, -1][full_states[:, -1] > self.path.period] -= self.path.period
        # full_states[:, -1][full_states[:, -1] <= 0] += self.path.period
        # states[:, -1] = full_states[:, -1]
        states[:, 4][states[:, 4] > np.pi] -= 2 * np.pi
        states[:, 4][states[:, 4] <= -np.pi] += 2 * np.pi

        return states

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        with tf.name_scope('compute_reward') as scope:
            delta_v_xs, v_ys, rs, delta_ys, delta_phis, xs = obs[:, 0], obs[:, 1], obs[:, 2], \
                                           obs[:, 3], obs[:, 4], obs[:, 5]
            steers, a_xs = actions[:, 0], actions[:, 1]
            punish_steer = -tf.square(steers)
            punish_a_x = -tf.square(a_xs)
            devi_v = -tf.square(delta_v_xs)
            devi_y = -tf.square(delta_ys)  # distence between current point and clostest point
            devi_phi = -tf.square(delta_phis)
            punish_yaw_rate = -tf.square(rs)
            # print('devi_y = ', devi_y)
            # print('devi_phi = ', devi_phi)
            # print('devi_v = ', devi_v)
            # print('punish_yaw_rate = ', punish_yaw_rate)
            # print('punish_steer = ', punish_steer)
            # exit()
            rewards = 0.01 * devi_v + 0.4 * devi_y + 0.1 * devi_phi + 0.2 * punish_yaw_rate + 5 * punish_steer + 0.05 * punish_a_x
            # rewards = 0.01 * devi_v + 1 * devi_y + 0.5 * devi_phi + 0.5 * punish_yaw_rate
        return rewards


class ReferencePath(object):
    def __init__(self):
        self.period = 1200.
        self.path_x = self.compute_x_point()
        self.path_y = self.compute_path_y(self.path_x)
        self.path_phi = self.compute_path_phi(self.path_x)


    def compute_path_y(self, x):
        y = np.zeros_like(x, dtype=np.float32)
        n = len(x)
        for _ in range(n):
            if x[_] <= 100.:
                y[_] = 1.875
        return y

    def compute_path_phi(self, x):
        deriv = np.zeros_like(x, dtype=np.float32)
        path_y = self.compute_path_y(x)
        n = len(x)
        for _ in range(n):
            if path_y[_] < 7.5:
                if x[_] <= 100.:
                    deriv[_] = 0
        return deriv

    def compute_y(self, x, delta_y):
        y_ref = self.compute_path_y(x)
        return delta_y + y_ref

    def compute_delta_y(self, x, y):
        y_ref = self.compute_path_y(x)
        return y - y_ref

    def compute_phi(self, x, delta_phi):
        phi_ref = self.compute_path_phi(x)
        phi = delta_phi + phi_ref
        phi[phi > np.pi] -= 2 * np.pi
        phi[phi <= -np.pi] += 2 * np.pi
        return phi

    def compute_x_point(self):
        x1 = np.linspace(50., 100., 1000).astype(np.float32)
        x2 = np.linspace(100., 105.625, 1000).astype(np.float32)
        x3 = np.linspace(105.625, 105.625, 1000).astype(np.float32)
        x = np.zeros((4000)).astype(np.float32)
        x[:1000] = x1
        x[1000: 2000] = x2
        x[2000:3000] = x3
        return x

    def indexs2points(self, indexs):
        indexs = tf.where(indexs >= 0, indexs, 0)
        indexs = tf.where(indexs < len(self.path_x), indexs, len(self.path_x)-1)
        points = tf.gather(self.path_x, indexs), \
                 tf.gather(self.path_y, indexs), \
                 tf.gather(self.path_phi, indexs)

        return points[0], points[1], points[2]

    def find_closest_point(self, xs, ys, ratio=1):
        path_len = len(self.path_x)
        reduced_idx = np.arange(0, path_len, ratio)
        reduced_len = len(reduced_idx)
        reduced_path_x, reduced_path_y = self.path_x[reduced_idx], self.path_y[reduced_idx]
        xs_tile = tf.tile(tf.reshape(xs, (-1, 1)), tf.constant([1, reduced_len]))
        ys_tile = tf.tile(tf.reshape(ys, (-1, 1)), tf.constant([1, reduced_len]))
        pathx_tile = tf.tile(tf.reshape(reduced_path_x, (1, -1)), tf.constant([len(xs), 1]))
        pathy_tile = tf.tile(tf.reshape(reduced_path_y, (1, -1)), tf.constant([len(xs), 1]))
        dist_array = tf.square(xs_tile - pathx_tile) + tf.square(ys_tile - pathy_tile)
        indexs = tf.argmin(dist_array, 1) * ratio
        return indexs, self.indexs2points(indexs)


class PathTrackingModel(object):  # all tensors
    def __init__(self, num_future_data=0, **kwargs):
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.obses = None
        self.actions = None
        self.veh_full_states = None
        self.num_future_data = num_future_data
        self.expected_vs = 2.
        self.path = ReferencePath()

    def reset(self, obses, veh_full_state): #obses 是delta_vx, v_y, r, delta_y, delta_phi, x
        self.obses = obses
        self.actions = None
        # self.veh_states = self._get_state(self.obses) #veh_states是vx, v_y, r, delta_y, delta_phi, x
        self.veh_full_states = veh_full_state

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
        actions = tf.stack([steer_norm * 1.2 * np.pi / 9, a_xs_norm * 3.], 1)
        self.actions = actions
        rewards = self.vehicle_dynamics.compute_rewards(self.obses, actions)
        self.veh_full_states = self.vehicle_dynamics.prediction(self.veh_full_states, actions,
                                                              self.base_frequency)
        v_xs, v_ys, rs, ys, phis, xs = self.veh_full_states[:, 0], self.veh_full_states[:, 1], self.veh_full_states[:, 2], \
                                       self.veh_full_states[:, 3], self.veh_full_states[:, 4], self.veh_full_states[:, 5]
        index, points = self.path.find_closest_point(xs, ys)
        path_x, path_y, path_phi = points[0], points[1], points[2]
        unsign_delta_y = tf.sqrt(tf.square(path_x - xs) + tf.square(path_y - ys))
        delta_ys = tf.where(ys > path_y, unsign_delta_y, - unsign_delta_y)
        delta_phis = phis - path_phi
        delta_phis = tf.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
        delta_phis = tf.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
        v_xs = tf.clip_by_value(v_xs, 1, 35)
        self.obses = tf.stack([v_xs - self.expected_vs, v_ys, rs, delta_ys, delta_phis, xs], axis=1)

        return self.obses, rewards, self.veh_full_states

