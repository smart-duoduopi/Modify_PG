from ini_state import Init_state
import numpy as np
from path_tracking_env import PathTrackingModel
import tensorflow as tf
from path_tracking_env import VehicleDynamics
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

    # def _get_obs(self, veh_state):
    #     v_xs, v_ys, rs, ys, phis, xs = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], veh_state[:, 3], veh_state[:, 4], veh_state[:, 5]
    #     lists_to_stack = [v_xs - self.expected_vs, v_ys, rs, ys, phis, xs]
    #     return np.stack(lists_to_stack, axis=1)

    # def _get_state(self, obses):
    #     v_xs, v_ys, rs, ys, phis, xs = obses[:, 0], obses[:, 1], obses[:, 2], obses[:, 3], obses[:, 4], obses[:, 5]
    #     lists_to_stack = [v_xs + self.expected_vs, v_ys, rs, ys, phis, xs]
    #     return np.stack(lists_to_stack, axis=1)

    # def _get_full_state(self, obses):
    #     delta_v_xs, v_ys, rs, delta_ys, delta_phis, xs = obses[:, 0], obses[:, 1], obses[:, 2], obses[:, 3], obses[:, 4], obses[:, 5]
    #     ys = self.vehicle_dynamics.path.compute_y(xs, delta_ys)
    #     phis = self.vehicle_dynamics.path.compute_phi(xs, delta_phis)
    #     lists_to_stack = [delta_v_xs + self.expected_vs, v_ys, rs, ys, phis, xs]
    #     return tf.stack(lists_to_stack, axis=1)

    def step(self, action):  # think of action is in range [-1, 1]
        steer_norm, a_xs_norm = action[:, 0], action[:, 1]
        action = tf.stack([steer_norm * 1.2 * np.pi / 9, a_xs_norm * 3.], 1)
        self.action = action
        action_tensor = tf.convert_to_tensor(self.action, dtype=tf.float32)
        reward = self.vehicle_dynamics.compute_rewards(self.obs, action_tensor).numpy()
        self.veh_full_state = self.vehicle_dynamics.simulation(self.veh_full_state, self.action, self.base_frequency)
        # print('self.veh_full_state = ', self.veh_full_state)
        v_xs, v_ys, rs, ys, phis, xs = self.veh_full_state[:, 0], self.veh_full_state[:, 1], self.veh_full_state[:, 2], \
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
    num = [[22], [40], [15000], [25000], [20000]]
    rr = []
    Env = PathTrackingEnv()
    for j in range(len(num)):
        reward = 0.
        Env.obs = train_set[num[j], :]
        Env.veh_full_state = train_set_full_state[num[j], :]
        # v_xs, v_ys, rs, ys, phis, xs = Env.veh_full_state[:, 0], Env.veh_full_state[:, 1], Env.veh_full_state[:, 2], \
        #                                Env.veh_full_state[:, 3], Env.veh_full_state[:, 4], Env.veh_full_state[:, 5]
        # print('v_xs = ', tf.convert_to_tensor(v_xs))
        # print('v_ys = ', tf.convert_to_tensor(v_ys))
        # print('rs = ', tf.convert_to_tensor(rs))
        # print('ys = ', tf.convert_to_tensor(ys))
        # print('phis = ', tf.convert_to_tensor(phis))
        # print('xs = ', tf.convert_to_tensor(xs))
        # index, points = Env.vehicle_dynamics.path.find_closest_point(xs, ys)
        # path_x, path_y, path_phi = points[0], points[1], points[2]
        # unsign_delta_y = tf.sqrt(tf.square(path_x - xs) + tf.square(path_y - ys))
        # if ys < 7.5:
        #     delta_ys = tf.where(ys > path_y, unsign_delta_y, - unsign_delta_y)
        # else:
        #     delta_ys = tf.where(xs > path_x, unsign_delta_y, - unsign_delta_y)
        # delta_phis = phis - path_phi
        # delta_phis = tf.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
        # delta_phis = tf.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
        # v_xs = tf.clip_by_value(v_xs, 1, 35)
        # Env.obs = tf.stack([v_xs - Env.expected_vs, v_ys, rs, delta_ys, delta_phis, xs], axis=1)
        xx = []
        yy = []
        for i in range(Env.interval_times):
            # print('Env.obs = ', tf.convert_to_tensor(Env.obs))
            # print('args.obs_scale', tf.convert_to_tensor(args.obs_scale))
            # print('Env.obs = ', tf.convert_to_tensor(Env.obs))
            action = policy(Env.obs * args.obs_scale)
            Env.obs, reward_new, x, y = Env.step(action)
            reward += reward_new
            xx.append(x)
            yy.append(y)

        if do == True:
            line_one_x = np.zeros((51))
            line_one_y = np.zeros((51))
            for _ in range (51):
                line_one_x[_] = 50 + _

            line_two_y = np.zeros((51))
            for _ in range(51):
                line_two_y[_] = 3.75

            line_three_y = np.zeros((51))
            for _ in range(51):
                line_three_y[_] = 7.5

            line_four_x = np.zeros((51))
            line_four_y = np.zeros((51))
            for _ in range(51):
                line_four_x[_] = 107.5 + _

            line_five_y = np.zeros((51))
            for _ in range(51):
                line_five_y[_] = 3.75

            line_six_y = np.zeros((51))
            for _ in range(51):
                line_six_y[_] = 7.5

            line_seven_x = np.zeros((51))
            line_seven_y = np.zeros((51))
            for _ in range(51):
                line_seven_x[_] = 100
                line_seven_y[_] = 7.5 + _

            line_eight_x = np.zeros((51))
            for _ in range(51):
                line_eight_x[_] = 103.75

            line_nine_x = np.zeros((51))
            for _ in range(51):
                line_nine_x[_] = 107.5

            line_ten_x = np.zeros((51))
            line_ten_y = np.zeros((51))
            for _ in range(51):
                line_ten_x[_] = 100
                line_ten_y[_] = 0 - _

            line_ele_x = np.zeros((51))
            for _ in range(51):
                line_ele_x[_] = 103.75

            line_twe_x = np.zeros((51))
            for _ in range(51):
                line_twe_x[_] = 107.5

            plt.plot(line_one_x, line_one_y, color='black', linestyle='-')
            plt.plot(line_one_x, line_two_y, color='black', linestyle='--')
            plt.plot(line_one_x, line_three_y, color='black', linestyle='-')
            plt.plot(line_four_x, line_four_y, color='black', linestyle='-')
            plt.plot(line_four_x, line_five_y, color='black', linestyle='--')
            plt.plot(line_four_x, line_six_y, color='black', linestyle='-')
            plt.plot(line_seven_x, line_seven_y, color='black', linestyle='-')
            plt.plot(line_eight_x, line_seven_y, color='black', linestyle='--')
            plt.plot(line_nine_x, line_seven_y, color='black', linestyle='-')
            plt.plot(line_ten_x, line_ten_y, color='black', linestyle='-')
            plt.plot(line_ele_x, line_ten_y, color='black', linestyle='--')
            plt.plot(line_twe_x, line_ten_y, color='black', linestyle='-')

            line_one_refx = np.zeros((51))
            line_one_refy = np.zeros((51))
            for _ in range (51):
                line_one_refx[_] = 50 + _
                line_one_refy[_] = 1.875
            plt.plot(line_one_refx, line_one_refy, color='blue', linestyle='--')

            line_two_refx = np.zeros((51))
            line_two_refy = np.zeros((51))
            for _ in range (51):
                line_two_refx[_] = 100 + 0.1125 * _
                line_two_refy[_] = 7.5 - np.sqrt(5.625 * 5.625 - np.square(line_two_refx[_] - 100.))
            plt.plot(line_two_refx, line_two_refy, color='blue', linestyle='--')

            line_three_refx = np.zeros((51))
            line_three_refy = np.zeros((51))
            for _ in range(51):
                line_three_refx[_] = 105.625
                line_three_refy[_] = 7.5 + _
            plt.plot(line_three_refx, line_three_refy, color='blue', linestyle='--')

            plt.plot(xx, yy, color='red', linestyle='-.')

            line_traffic_light_refx = np.zeros((51))
            line_traffic_light_refy = np.zeros((51))
            for _ in range(51):
                line_traffic_light_refx[_] = 100
                line_traffic_light_refy[_] = 0 + _ * 0.075
            # plt.xlim(87.5, 120)
            # plt.ylim(-12.5, 20)
            plt.axis('equal')
            plt.show()
            print('reward = ', reward)
        rr.append(reward)
    reward = np.mean(rr)
    # io.savemat('x_actual_DADP_48_12X4_sy_2000.mat', {'array': x_actual})
    # io.savemat('y_actual_DADP_48_12X4_sy_2000.mat', {'array': y_actual})
    # io.savemat('y_ref_DADP_48_12X4_sy_2000.mat', {'array': y_ref})

    return reward
