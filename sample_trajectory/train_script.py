import codecs
import os
import ray
import time
import logging
import argparse
import datetime
import numpy as np
import tensorflow as tf
from sample_trajectory.ampc import AMPCLearner
#from dmpc import DMPC
from sample_trajectory.policy import PolicyWithQs
from sample_trajectory.evaluation import evaluation
from sample_trajectory.ini_state import Init_state
import datetime
import matplotlib.pyplot as plt
import random


tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


def built_DADP_parser():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--code_mode', default='evaluate', help='train or evaluate')
    parser.add_argument('--algorithm_mode', default='ADP', help='ADP')
    parser.add_argument('--max_iteration', type=int, default=3000)
    # number_init_state
    parser.add_argument('--num_state', type=int, default=500)
    # learner
    parser.add_argument('--prediction_horizon', type=int, default=100)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)
    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    # policy and model
    parser.add_argument('--obs_dim', type=int, default=4)
    parser.add_argument('--act_dim', type=int, default=1)
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='tanh')
    parser.add_argument('--policy_lr_schedule', type=list, default=1e-5)
    # preprocessor
    # 为将状态量对reward的影响拉到同一维度，做归一化处理。
    # delta_v_xs, v_ys, rs, delta_ys, delta_phis, xs
    parser.add_argument('--obs_scale', type=list, default=[0.3, 1., 0.5, 1.])
    # 否则有的状态数值很大，对训练以及评估存在问题
    # # parser.add_argument('--obs_scale', type=list, default=[1., 1., 2., 1., 2.4, 1/120])
    # parser.add_argument('--obs_scale', type=list, default=[0.5, 2., 1., 0.5, 1., 1 / 108])
    # # parser.add_argument('--eva_scale', type=list, default=[1., 2., 1., 2.4, 1/1200])
    # # parser.add_argument('--loss_scale', type=list, default=[1., 1., 1., 1.])
    # # parser.add_argument('--rew_scale', type=float, default=0.01)
    #
    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = '../adp/results/experiment-{time}'.format(time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)
    return parser.parse_args()


def main():
    args = built_DADP_parser()
    if args.algorithm_mode == 'ADP':
        log_dir = "../adp/summary_writer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = tf.summary.create_file_writer(log_dir)
        learner = AMPCLearner(PolicyWithQs, args=args)

        if args.code_mode == 'train':
            # ini = Init_state(args.num_state)
            # train_set_obs_x = ini.reset()
            # np.save('train_set.npy', train_set_obs_x)
            # print('success')
            # exit()
            train_set_obs_x = np.load('train_set.npy')
            learner.get_batch_data(train_set_obs_x)
            start_time = time.time()
            for ite in range(args.max_iteration):
                policy_loss = learner.policy_forward_and_backward()
                print('policy_loss = ', policy_loss)
                with summary_writer.as_default():
                    tf.summary.scalar('loss', policy_loss, ite, None)
                if ite == args.max_iteration - 1:
                    learner.policy_with_value.save_weights(args.model_dir, ite)
            end_time = time.time()
            print('time = ', end_time - start_time, 's')
        elif args.code_mode == 'evaluate':
            # ini = Init_state(args.num_state
            train_set = np.load('train_set.npy')
            learner.policy_with_value.load_weights('../adp/results/experiment-2022-02-25-12-46-33/models', 2999)
            evaluation(args, learner.policy_with_value.policy, train_set, True)


if __name__ == '__main__':
    main()
