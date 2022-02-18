#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf

from model import MLPNet

NAME2MODELCLS = dict([('MLP', MLPNet)])


class PolicyWithQs(tf.Module):
    import tensorflow as tf

    def __init__(self, obs_dim, act_dim,
                 policy_model_cls, policy_num_hidden_layers, policy_num_hidden_units, policy_hidden_activation,
                 policy_out_activation, policy_lr_schedule, **kwargs):
        super().__init__()
        policy_model_cls = NAME2MODELCLS[policy_model_cls]
        self.policy = policy_model_cls(obs_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                       policy_hidden_activation, act_dim, name='policy',
                                       output_activation=policy_out_activation)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='policy_adam_opt')
        self.models = (self.policy,)
        self.optimizers = (self.policy_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, grads):
        policy_grad = grads
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))

    @tf.function
    def compute_action(self, obs):
        logits = self.policy(obs)
        return logits

