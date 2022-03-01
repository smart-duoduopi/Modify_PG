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
        self.action = None
        self.interval_times = 200
        self.expected_vs = 20.

    def step(self, action):  # think of action is in range [-1, 1]
        steer_norm = action[:, 0]
        self.action = steer_norm * 1.2 * np.pi / 9
        action_tensor = tf.convert_to_tensor(self.action, dtype=tf.float32)
        reward = self.vehicle_dynamics.compute_rewards(self.obs, action_tensor).numpy()
        self.obs = self.vehicle_dynamics.prediction(self.obs, self.action)
        v_ys, rs, delta_ys, delta_phis, xs = self.obs[:, 0], self.obs[:, 1], self.obs[:, 2],\
                                       self.obs[:, 3], self.obs[:, 4],
        delta_phis = tf.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
        delta_phis = tf.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
        obses_scale = tf.stack([v_ys, rs, delta_ys, delta_phis], axis=1)
        self.obs = tf.stack([v_ys, rs, delta_ys, delta_phis, xs], axis=1)
        return obses_scale, self.obs, reward, xs, delta_ys

def evaluation(args, policy, train_set, do):
    num = [[20], [40], [60], [80], [50]]
    # num = [[0]]
    rr = []
    Env = PathTrackingEnv()
    for j in range(len(num)):
        reward = 0.
        Env.obs = train_set[num[j], :]
        obses_scale = Env.obs[:, 0:4]
        xx = []
        yy = []
        for i in range(Env.interval_times):
            action = policy(obses_scale * args.obs_scale)
            obses_scale, Env.obs, reward_new, x, y = Env.step(action)
            reward += reward_new
            xx.append(x)
            yy.append(y)

        if do == True:
            # line_one_x = np.zeros((100))
            # line_one_y = np.zeros((100))
            # for _ in range(100):
            #     line_one_x[_] = 50 + _
            #
            # line_two_y = np.zeros((100))
            # for _ in range(100):
            #     line_two_y[_] = 3.75
            #
            # line_three_y = np.zeros((100))
            # for _ in range(100):
            #     line_three_y[_] = 7.5
            #
            # plt.plot(line_one_x, line_one_y, color='black', linestyle='-')
            # plt.plot(line_one_x, line_two_y, color='black', linestyle='--')
            # plt.plot(line_one_x, line_three_y, color='black', linestyle='-')
            #
            # line_one_refx = np.zeros((100))
            # line_one_refy = np.zeros((100))
            # for _ in range(100):
            #     line_one_refx[_] = 50 + _
            #     line_one_refy[_] = 1.875
            # plt.plot(line_one_refx, line_one_refy, color='blue', linestyle='--')
            plt.plot(xx, yy, color='red', linestyle='-.')
            plt.ylim(-10, 10)
            # plt.axis('equal')
            plt.show()
            print('reward = ', reward)
        rr.append(reward)
    reward = np.mean(rr)

    return reward
