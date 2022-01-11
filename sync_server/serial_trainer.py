from sync_server.model_define import System_define
import copy
import numpy as np

class serial_trainer():
    def __init__(self, alg):
        self.algs = alg
        self.iteration = 0
        self.networks = System_define()
        self.P_old = copy.deepcopy(self.networks.P)

    def _set_algs(self):
        weights = self.networks.state_dict()  # 获得中心网络参数
        self.algs.load_state_dict(weights)  # 每个learner同步参数

    def step(self):
        grads = self.algs.cal_grad()
        self.networks.update_P(grads)
        weights = self.networks.state_dict()  # 把中心网络的参数放在底层内存里面
        self.algs.load_state_dict(weights)  # 更新learner参数
        self.iteration += 1

    def train(self):
        while True:
            self.step()
            if np.abs(np.sum(self.networks.P - self.P_old)) < 0.0001:
                print('successful')
                print('ite =', self.iteration)
                break
            self.P_old = copy.deepcopy(self.networks.P)
            if self.iteration > 2000:
                print('warning')
                break