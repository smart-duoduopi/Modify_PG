import logging
import numpy as np
from sample_trajectory.path_tracking_env import PathTrackingModel
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
        self.model = PathTrackingModel(**vars(self.args))

    def get_batch_data(self, batch_data):
        self.batch_data = batch_data

    def model_rollout_for_policy_update(self, start_obses):
        self.model.reset(start_obses)
        rewards_sum = self.tf.zeros((start_obses.shape[0],))
        # print('start_obses = ', start_obses)
        # obses_scale = start_obses
        for _ in range(self.args.prediction_horizon):
            # print('obses_scale = ', obses_scale)
            obses_scale = start_obses * self.args.obs_scale
            actions = self.policy_with_value.compute_action(obses_scale)
            obses, rewards = self.model.rollout_out(actions)
            # print('reward = ', rewards_sum)
            rewards_sum += rewards

            obses_scale = obses[:, 0:4]
        # print('##########################')
            # obses 是 v_y, r, delta_y, delta_phi
        # exit()
        policy_loss = self.tf.reduce_mean(rewards_sum)
        return policy_loss


    def policy_forward_and_backward(self):
        mb_obs = self.batch_data.copy()
        # TensorFlow 为自动微分提供了 tf.GradientTape API ，根据某个函数的输入变量来计算它的导数。
        # Tensorflow 会把 ‘tf.GradientTape’ 上下文中执行的所有操作都记录在一个磁带上 (“tape”)。
        # 然后基于这个磁带和每次操作产生的导数，
        # 用反向微分法（“reverse mode differentiation”）来计算这些被“记录在案”的函数的导数。
        with self.tf.GradientTape() as tape:
            policy_loss = self.model_rollout_for_policy_update(mb_obs)
        with self.tf.name_scope('policy_gradient') as scope: # 用于tensorboard可视化时，将部分底层的模块归结于一个里面。类似于simulink中的可视化的美化功能。为了不那么凌乱
            policy_gradient = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights)# 这里的policy是model.py函数中的MLPNet
        # policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(policy_gradient,
        #                                                                     self.args.gradient_clip_norm)
        self.policy_with_value.apply_gradients(policy_gradient)
        return policy_loss