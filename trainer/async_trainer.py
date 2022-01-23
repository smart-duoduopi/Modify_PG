from model.model_define import System_define
from trainer.task_pool import TaskPool
import copy
import ray
import numpy as np

class async_trainer():
    def __init__(self, alg):
        self.algs = alg
        self.iteration = 0
        self.networks = System_define()
        # print(self.networks.P)
        self.learn_tasks = TaskPool()  # 创建learner的任务管理的类
        # self._set_algs()
        # self.start_time = time.time()
        # self.recode_error = []
        # self.max_iteration = 2000
        self.P_old = copy.deepcopy(self.networks.P)

    def _set_algs(self):
        weights = self.networks.state_dict()  # 获得中心网络参数
        for alg in self.algs:
            alg.load_state_dict.remote(weights)  # 每个learner同步参数
            self.learn_tasks.add(alg, alg.cal_grad.remote())  # 用采样结果给learner添加计算梯度的任务
        # self.algs.load_state_dict(weights)  # 每个learner同步参数
        # self.learn_tasks.add(alg, alg.cal_grad.remote())  # 用采样结果给learner添加计算梯度的任务

    def step(self):
        for alg, objID in self.learn_tasks.completed():
            grads = ray.get(objID)
        # grads = self.algs.cal_grad()
            self.networks.update_P(grads)
        # weights = self.networks.state_dict()
            weights = ray.put(self.networks.state_dict())  # 把中心网络的参数放在底层内存里面
            alg.load_state_dict.remote(weights)  # 更新learner参数
        # self.algs.load_state_dict(weights)
            self.learn_tasks.add(alg, alg.cal_grad.remote())  # 将完成了的learner重新算梯度
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