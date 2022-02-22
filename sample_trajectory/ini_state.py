import numpy as np
from sample_trajectory.path_tracking_env import VehicleDynamics
from collections import deque


class Init_state(object):
    def __init__(self, num_agent=1, **kwargs):
        # veh_state = obs: v_ys, rs, delta_ys, delta_phis, xs
        # obs:v_ys, rs, delta_ys, delta_phis, xs, future_delta_ys1,..., future_delta_ysn,
        #         #      future_delta_phis1,..., future_delta_phisn
        # veh_full_state:  v_ys, rs, ys, phis, xs
        self.obs = None
        self.num_agent = num_agent
        self.expect_v = 20.

    def _get_obs(self, veh_state):
        v_ys, rs, ys, phis = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], veh_state[:, 3]
        lists_to_stack = [v_ys, rs, ys, phis]
        return np.stack(lists_to_stack, axis=1)

    def reset(self, **kwargs):
        init_x = np.random.uniform(0., 100., (self.num_agent,)).astype(np.float32)
        init_delta_y = np.random.normal(0, 0.7, (self.num_agent,)).astype(np.float32)  # delta_y \in [-2, 2]
        init_delta_phi = np.random.normal(0, np.pi / 9, (self.num_agent,)).astype(np.float32)  # delta_phi \in [-1, 1]
        init_delta_v_x = np.random.normal(0, 0.3, (self.num_agent,)).astype(np.float32)  # delta_vx \in [-1, 1]
        beta = np.random.normal(0, 0.15, (self.num_agent,)).astype(np.float32)
        init_v_y = (init_delta_v_x + self.expect_v) * np.tan(beta)
        init_r = np.random.normal(0, 0.3, (self.num_agent,)).astype(np.float32)  # r \in [-1, 1]
        self.obs = np.stack([init_v_y, init_r, init_delta_y, init_delta_phi, init_x], 1)
        return self.obs
