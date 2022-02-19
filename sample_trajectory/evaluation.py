from sample_trajectory.ini_state import Init_state
import numpy as np
from sample_trajectory.path_tracking_env import PathTrackingModel
import tensorflow as tf
from sample_trajectory.path_tracking_env import VehicleDynamics
import matplotlib.pyplot as plt
from scipy import io

class PathTrackingEnv(object):
    def __init__(self, **kwargs):
        self.vehicle_dynamics = VehicleDynamics()
        self.obs = None
        self.veh_full_state = None
        self.action = None
        self.base_frequency = 10
        self.interval_times = 200
        self.expected_vs = 2.

    def step(self, action):  # think of action is in range [-1, 1]
        steer_norm, a_xs_norm = action[:, 0], action[:, 1]
        action = tf.stack([steer_norm * 1.2 * np.pi / 9, a_xs_norm * 3.], 1)
        self.action = action
        action_tensor = tf.convert_to_tensor(self.action, dtype=tf.float32)
        reward = self.vehicle_dynamics.compute_rewards(self.obs, action_tensor).numpy()
        self.veh_full_state = self.vehicle_dynamics.simulation(self.veh_full_state, self.action, self.base_frequency)
        v_xs, v_ys, rs, ys, phis, xs = self.veh_full_state[:, 0], self.veh_full_state[:, 1], self.veh_full_state[:, 2],\
                                       self.veh_full_state[:, 3], self.veh_full_state[:, 4], self.veh_full_state[:, 5]
        index, points = self.vehicle_dynamics.path.find_closest_point(xs, ys)
        path_x, path_y, path_phi = points[0], points[1], points[2]
        unsign_delta_y = tf.sqrt(tf.square(path_x - xs) + tf.square(path_y - ys))
        if path_y >= 7.5:
            delta_ys = tf.where(xs > path_x, unsign_delta_y, - unsign_delta_y)
        else:
            delta_ys = tf.where(ys > path_y, unsign_delta_y, - unsign_delta_y)
        delta_phis = phis - path_phi
        delta_phis = tf.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
        delta_phis = tf.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
        v_xs = tf.clip_by_value(v_xs, 1, 35)
        self.obs = tf.stack([v_xs - self.expected_vs, v_ys, rs, delta_ys, delta_phis, xs], axis=1)
        x = self.veh_full_state[:, -1]
        y = self.veh_full_state[:, 3]
        return self.obs, reward, x, y

def evaluation(args, policy, train_set, train_set_full_state, do):
    num = [[20], [40], [60], [80], [50]]
    rr = []
    Env = PathTrackingEnv()
    for j in range(len(num)):
        reward = 0.
        Env.obs = train_set[num[j], :]
        Env.veh_full_state = train_set_full_state[num[j], :]
        xx = []
        yy = []
        for i in range(Env.interval_times):
            action = policy(Env.obs * args.obs_scale)
            Env.obs, reward_new, x, y = Env.step(action)
            reward += reward_new
            xx.append(x)
            yy.append(y)

        if do == True:
            line_one_x = np.zeros((100))
            line_one_y = np.zeros((100))
            for _ in range(100):
                line_one_x[_] = 50 + _

            line_two_y = np.zeros((100))
            for _ in range(100):
                line_two_y[_] = 3.75

            line_three_y = np.zeros((100))
            for _ in range(100):
                line_three_y[_] = 7.5

            plt.plot(line_one_x, line_one_y, color='black', linestyle='-')
            plt.plot(line_one_x, line_two_y, color='black', linestyle='--')
            plt.plot(line_one_x, line_three_y, color='black', linestyle='-')

            line_one_refx = np.zeros((100))
            line_one_refy = np.zeros((100))
            for _ in range(100):
                line_one_refx[_] = 50 + _
                line_one_refy[_] = 1.875
            plt.plot(line_one_refx, line_one_refy, color='blue', linestyle='--')
            plt.plot(xx, yy, color='red', linestyle='-.')

            plt.axis('equal')
            plt.show()
            print('reward = ', reward)
        rr.append(reward)
    reward = np.mean(rr)

    return reward
