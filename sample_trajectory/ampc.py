#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/11
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ampc.py
# =====================================

import logging
import numpy as np
from path_tracking_env import PathTrackingModel
import matplotlib.pyplot as plt
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AMPCLearner(object):
    import tensorflow as tf

    def __init__(self, policy_cls, args):
        self.args = args
        self.policy_with_value = policy_cls(**vars(self.args))
        self.batch_data = None
        self.batch_full_data = None
        self.model = PathTrackingModel(**vars(self.args))

    def get_batch_data(self, batch_data, batch_full_data):
        self.batch_data = batch_data
        self.batch_full_data = batch_full_data

    def model_rollout_for_policy_update(self, start_obses, veh_full_state):
        # print('start_obses = ', start_obses)
        # print('veh_full_state = ', veh_full_state)
        # exit()
        self.model.reset(start_obses, veh_full_state)
        rewards_sum = self.tf.zeros((start_obses.shape[0],))
        obses = start_obses

        plt.axis([90, 110, 0, 10])
        plt.ion()
        for _ in range(self.args.prediction_horizon):
            obses = obses * self.args.obs_scale
            # 对obs scale之后输入网络计算动作
            veh_state = tf.convert_to_tensor(veh_full_state)
            # print('veh_state = ', veh_state)
            # print('veh_state[0] = ', veh_state[:, -1])
            # exit()

            line_one_x = np.zeros((51))
            line_one_y = np.zeros((51))
            for _ in range(51):
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
            for _ in range(51):
                line_one_refx[_] = 50 + _
                line_one_refy[_] = 1.875
            plt.plot(line_one_refx, line_one_refy, color='blue', linestyle='--')

            line_two_refx = np.zeros((51))
            line_two_refy = np.zeros((51))
            for _ in range(51):
                line_two_refx[_] = 100 + 0.1125 * _
                line_two_refy[_] = 7.5 - np.sqrt(5.625 * 5.625 - np.square(line_two_refx[_] - 100.))
            plt.plot(line_two_refx, line_two_refy, color='blue', linestyle='--')

            line_three_refx = np.zeros((51))
            line_three_refy = np.zeros((51))
            for _ in range(51):
                line_three_refx[_] = 105.625
                line_three_refy[_] = 7.5 + _
            plt.plot(line_three_refx, line_three_refy, color='blue', linestyle='--')

            line_traffic_light_refx = np.zeros((51))
            line_traffic_light_refy = np.zeros((51))
            for _ in range(51):
                line_traffic_light_refx[_] = 100
                line_traffic_light_refy[_] = 0 + _ * 0.075
            x = veh_state[:, -1]
            y = veh_state[:, 3]
# zhuangtai, dongzuo, reward,
            x_mowei = x + 1 * np.cos(veh_state[:, 4])
            y_mowei = y + 1 * np.sin(veh_state[:, 4])
            plt.plot([x, x_mowei], [y, y_mowei])
            plt.pause(1)
            # plt.scatter(x, y, color='red')
            # plt.axis('equal')
            # plt.show()
            actions = self.policy_with_value.compute_action(obses)
            obses, rewards, veh_full_state = self.model.rollout_out(actions)
            rewards_sum += rewards
            # obses 是 delta_vx, v_y, r, delta_y, delta_phi, x
        policy_loss = - self.tf.reduce_mean(rewards_sum)
        return policy_loss


    def policy_forward_and_backward(self):
        mb_obs = self.batch_data.copy()
        veh_full_state = self.batch_full_data.copy()
        # TensorFlow 为自动微分提供了 tf.GradientTape API ，根据某个函数的输入变量来计算它的导数。
        # Tensorflow 会把 ‘tf.GradientTape’ 上下文中执行的所有操作都记录在一个磁带上 (“tape”)。
        # 然后基于这个磁带和每次操作产生的导数，
        # 用反向微分法（“reverse mode differentiation”）来计算这些被“记录在案”的函数的导数。
        with self.tf.GradientTape() as tape:
            policy_loss = self.model_rollout_for_policy_update(mb_obs, veh_full_state)
        with self.tf.name_scope('policy_gradient') as scope: # 用于tensorboard可视化时，将部分底层的模块归结于一个里面。类似于simulink中的可视化的美化功能。为了不那么凌乱
            policy_gradient = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights)# 这里的policy是model.py函数中的MLPNet
        policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(policy_gradient,
                                                                            self.args.gradient_clip_norm)
        self.policy_with_value.apply_gradients(policy_gradient)
        return policy_loss