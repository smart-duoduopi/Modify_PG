#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: train_script.py
# =====================================
# -*- coding: utf-8 -*-
import codecs
import os
import ray
import time
import logging
import argparse
import datetime
import numpy as np
import tensorflow as tf
from dadp_diff_learner_rate import DADP
from dadp_diff_learner_rate_async import Async_DADP
from ampc import AMPCLearner
#from dmpc import DMPC
from policy import PolicyWithQs
from evaluation import evaluation
from ini_state import Init_state
import datetime
import matplotlib.pyplot as plt
import random
#import sys
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


def built_DADP_parser():
    parser = argparse.ArgumentParser()
    # ADMM
    parser.add_argument('--ADMM_max_iteration', type=int, default=200)
    parser.add_argument('--rou', type=float, default=2)
    parser.add_argument('--eps_abs', type=float, default=0.001)
    parser.add_argument('--eps_rel', type=float, default=0.001)
    # mode
    parser.add_argument('--code_mode', default='train', help='train or evaluate')
    parser.add_argument('--algorithm_mode', default='ADP', help='DADP or ADP')
    # number_init_state
    parser.add_argument('--num_state', type=int, default=10000)
    # learner
    parser.add_argument('--prediction_horizon', type=int, default=60)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)
    parser.add_argument('--max_iteration', type=float, default=100)
    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    # policy and model
    parser.add_argument('--obs_dim', type=int, default=6)
    parser.add_argument('--act_dim', type=int, default=2)
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='tanh')
    parser.add_argument('--policy_lr_schedule', type=list, default=3e-4)
    parser.add_argument('--state_lr_schedule', type=list, default=3e-2)
    # preprocessor
    # 为将状态量对reward的影响拉到同一维度，做归一化处理。
    # 否则有的状态数值很大，对训练以及评估存在问题
    # parser.add_argument('--obs_scale', type=list, default=[1., 1., 2., 1., 2.4, 1/120])
    parser.add_argument('--obs_scale', type=list, default=[0.5, 2., 1., 0.5, 1., 1 / 108])
    # parser.add_argument('--eva_scale', type=list, default=[1., 2., 1., 2.4, 1/1200])
    # parser.add_argument('--loss_scale', type=list, default=[1., 1., 1., 1.])
    # parser.add_argument('--rew_scale', type=float, default=0.01)
    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = '../dadp/adp/results/adp/experiment-{time}'.format(time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    DADP_results_dir = '../dadp/paper/results/dadp/experiment-{time}'.format(time=time_now)
    parser.add_argument('--DADP_result_dir', type=str, default=DADP_results_dir)
    parser.add_argument('--DADP_log_dir', type=str, default=DADP_results_dir + '/DADP_log_dir')
    parser.add_argument('--DADP_model_dir', type=str, default=DADP_results_dir + '/DADP_model_dir')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)
    return parser.parse_args()


def main():
    args = built_DADP_parser()
    if args.algorithm_mode == 'ADP':
        log_dir = "../dadp/adp/summary_writer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = tf.summary.create_file_writer(log_dir)
        learner = AMPCLearner(PolicyWithQs, args=args)

        if args.code_mode == 'train':
            # ini = Init_state(args.num_state)
            # train_set_obs, train_set_full_state = ini.reset()
            # np.save('train_set_2.npy', train_set_obs)
            # np.save('train_set_5.npy', train_set_full_state)
            # print('train_set = ', train_set_obs)
            # print('train_set_full_state = ', train_set_full_state)
            # exit()
            # train_set = np.load('train_set_1.npy')
            set_1 = np.load('train_set_1.npy')
            set_2 = np.load('train_set_2.npy')
            set_3 = np.load('train_set_3.npy')
            # print('check')
            # exit()
            # train_set_obs_state = np.zeros((30000, 6)).astype(np.float32)
            # train_set_obs_state[:10000, :] = set_1
            # train_set_obs_state[10000: 20000, :] = set_2
            # train_set_obs_state[20000:30000, :] = set_3
            set_4 = np.load('train_set_4.npy')
            set_5 = np.load('train_set_5.npy')
            set_6 = np.load('train_set_6.npy')
            # train_set_full_state = np.zeros((150, 6)).astype(np.float32)
            # train_set_full_state[:50, :] = set_4
            # train_set_full_state[50: 100, :] = set_5
            # train_set_full_state[100:150, :] = set_6
            # line_one_x = np.zeros((51))
            # line_one_y = np.zeros((51))
            # for _ in range(51):
            #     line_one_x[_] = 50 + _
            #
            # line_two_y = np.zeros((51))
            # for _ in range(51):
            #     line_two_y[_] = 3.75
            #
            # line_three_y = np.zeros((51))
            # for _ in range(51):
            #     line_three_y[_] = 7.5
            #
            # line_four_x = np.zeros((51))
            # line_four_y = np.zeros((51))
            # for _ in range(51):
            #     line_four_x[_] = 107.5 + _
            #
            # line_five_y = np.zeros((51))
            # for _ in range(51):
            #     line_five_y[_] = 3.75
            #
            # line_six_y = np.zeros((51))
            # for _ in range(51):
            #     line_six_y[_] = 7.5
            #
            # line_seven_x = np.zeros((51))
            # line_seven_y = np.zeros((51))
            # for _ in range(51):
            #     line_seven_x[_] = 100
            #     line_seven_y[_] = 7.5 + _
            #
            # line_eight_x = np.zeros((51))
            # for _ in range(51):
            #     line_eight_x[_] = 103.75
            #
            # line_nine_x = np.zeros((51))
            # for _ in range(51):
            #     line_nine_x[_] = 107.5
            #
            # line_ten_x = np.zeros((51))
            # line_ten_y = np.zeros((51))
            # for _ in range(51):
            #     line_ten_x[_] = 100
            #     line_ten_y[_] = 0 - _
            #
            # line_ele_x = np.zeros((51))
            # for _ in range(51):
            #     line_ele_x[_] = 103.75
            #
            # line_twe_x = np.zeros((51))
            # for _ in range(51):
            #     line_twe_x[_] = 107.5
            #
            # plt.plot(line_one_x, line_one_y, color='black', linestyle='-')
            # plt.plot(line_one_x, line_two_y, color='black', linestyle='--')
            # plt.plot(line_one_x, line_three_y, color='black', linestyle='-')
            # plt.plot(line_four_x, line_four_y, color='black', linestyle='-')
            # plt.plot(line_four_x, line_five_y, color='black', linestyle='--')
            # plt.plot(line_four_x, line_six_y, color='black', linestyle='-')
            # plt.plot(line_seven_x, line_seven_y, color='black', linestyle='-')
            # plt.plot(line_eight_x, line_seven_y, color='black', linestyle='--')
            # plt.plot(line_nine_x, line_seven_y, color='black', linestyle='-')
            # plt.plot(line_ten_x, line_ten_y, color='black', linestyle='-')
            # plt.plot(line_ele_x, line_ten_y, color='black', linestyle='--')
            # plt.plot(line_twe_x, line_ten_y, color='black', linestyle='-')
            #
            # line_one_refx = np.zeros((51))
            # line_one_refy = np.zeros((51))
            # for _ in range(51):
            #     line_one_refx[_] = 50 + _
            #     line_one_refy[_] = 1.875
            # plt.plot(line_one_refx, line_one_refy, color='blue', linestyle='--')
            #
            # line_two_refx = np.zeros((51))
            # line_two_refy = np.zeros((51))
            # for _ in range(51):
            #     line_two_refx[_] = 100 + 0.1125 * _
            #     line_two_refy[_] = 7.5 - np.sqrt(5.625 * 5.625 - np.square(line_two_refx[_] - 100.))
            # plt.plot(line_two_refx, line_two_refy, color='blue', linestyle='--')
            #
            # line_three_refx = np.zeros((51))
            # line_three_refy = np.zeros((51))
            # for _ in range(51):
            #     line_three_refx[_] = 105.625
            #     line_three_refy[_] = 7.5 + _
            # plt.plot(line_three_refx, line_three_refy, color='blue', linestyle='--')
            #
            # line_traffic_light_refx = np.zeros((51))
            # line_traffic_light_refy = np.zeros((51))
            # for _ in range(51):
            #     line_traffic_light_refx[_] = 100
            #     line_traffic_light_refy[_] = 0 + _ * 0.075
            #
            # set_4 = np.load('train_set_4.npy')
            # set_5 = np.load('train_set_5.npy')
            # set_6 = np.load('train_set_6.npy')
            # train_set_full_state = np.zeros((30000, 6)).astype(np.float32)
            # train_set_full_state[:10000, :] = set_4
            # train_set_full_state[10000: 20000, :] = set_5
            # train_set_full_state[20000:30000, :] = set_6
            # x = np.zeros((30000)).astype(np.float32)
            # y = np.zeros((30000)).astype(np.float32)
            # x_mowei = np.zeros((30000)).astype(np.float32)
            # y_mowei = np.zeros((30000)).astype(np.float32)
            # for _ in range(30000):
            #     x[_] = train_set_full_state[_, -1]
            #     y[_] = train_set_full_state[_, 3]
            #
            # for _ in range(30000):
            #     x_mowei[_] = x[_] + 1 * np.cos(train_set_full_state[_, 4])
            #     y_mowei[_] = y[_] + 1 * np.sin(train_set_full_state[_, 4])
            #
            #
            # plt.plot([x,x_mowei], [y, y_mowei])
            #
            #
            # plt.scatter(x, y, color='red')
            # plt.axis('equal')
            # plt.show()
            # exit()
            # np.save('train_set_total.npy', train_set)
            # np.save('train_set_obs_state.npy', train_set_obs_state)
            data_num = 1
            num = random.choice(range(0, 10000 - data_num))
            a = set_1[num:num + data_num]
            b = set_2[num:num + data_num]
            c = set_3[num:num + data_num]
            d = set_4[num:num + data_num]
            e = set_5[num:num + data_num]
            f = set_6[num:num + data_num]
            train_set = np.zeros((data_num * 3, 6)).astype(np.float32)
            train_set[:data_num, :] = a
            train_set[data_num: 2 * data_num, :] = b
            train_set[2 * data_num:3 * data_num, :] = c
            train_set_full_state = np.zeros((data_num * 3, 6)).astype(np.float32)
            train_set_full_state[:data_num, :] = d
            train_set_full_state[data_num: 2 * data_num, :] = e
            train_set_full_state[2 * data_num: 3 * data_num, :] = f
            learner.get_batch_data(train_set, train_set_full_state)
            np.save('train_obs_batch.npy', train_set)
            np.save('train_full_state_batch.npy', train_set_full_state)
            start_time = time.time()
            for ite in range(args.ADMM_max_iteration):
                policy_loss = learner.policy_forward_and_backward()
                print('policy_loss = ', policy_loss)
                with summary_writer.as_default():
                    tf.summary.scalar('loss', policy_loss, ite, None)
                if ite == args.ADMM_max_iteration - 1:
                    learner.policy_with_value.save_weights(args.model_dir, ite)
            end_time = time.time()
            print('time = ', end_time - start_time, 's')
        elif args.code_mode == 'evaluate':
            train_set = np.load('train_obs_batch.npy')
            train_set_full_state = np.load('train_full_state_batch.npy')
            learner.policy_with_value.load_weights('D:/完成的一些ppt和文档/小论文/DADP/dadp/adp/results/adp/experiment-2021-09-23-12-51-19/models', 199)
            evaluation(args, learner.policy_with_value.policy, train_set, train_set_full_state, True)
    elif args.algorithm_mode == 'DADP':
        ray.init()
        train_set = np.load('train_set.npy')
        num = [[1], [10], [20], [30], [40]]
        ini_states = np.zeros((len(num), args.obs_dim), dtype=np.float32)
        for jj in range(len(num)):
            ini_states[jj, :] = train_set[num[jj], :]

        learner = Async_DADP(train_set, args)
        if args.code_mode == 'train':
            start_time = time.time()
            learner.train()
            end_time = time.time()
            print('total_time =')
            print(end_time - start_time)
        elif args.code_mode == 'evaluate':
            train_set = np.load('train_set.npy')
            learner.load_weights('F:/完成的一些ppt和文档/DADP/dadp/paper_data/paper/results/dadp/experiment-2021-09-23-10-50-04/DADP_model_dir', 99)
            evaluation(args, learner.all_parameter.policy.params[0], learner.model.vehicle_dynamics, train_set, True)


if __name__ == '__main__':
    main()
