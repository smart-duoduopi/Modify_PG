from model.model_define import System_define
from trainer.task_pool import TaskPool
import copy
import ray
import numpy as np
import matplotlib.pyplot as plt

class async_trainer_modify():
    def __init__(self, alg, num):
        self.algs = alg
        self.iteration = 0
        self.networks = System_define()
        self.learn_tasks = TaskPool()  # 创建learner的任务管理的类
        self.P_old = copy.deepcopy(self.networks.P)
        self.P_main_recode_0 = []
        self.P_main_recode_1 = []
        self.P_main_recode_2 = []
        self.P_main_recode_3 = []
        self.P_sub_recode_0 = []
        self.P_sub_recode_1 = []
        self.P_sub_recode_2 = []
        self.P_sub_recode_3 = []
        self.num = num

    def _set_algs(self):
        weights = self.networks.state_dict()  # 获得中心网络参数
        for alg in self.algs:
            alg.load_state_dict.remote(weights)  # 每个learner同步参数
            self.learn_tasks.add(alg, alg.cal_grad.remote())  # 用采样结果给learner添加计算梯度的任务

    def sync(self):
        weights = self.networks.state_dict()  # 获得中心网络参数
        for alg in self.algs:
            alg.load_state_dict.remote(weights)  # 每个learner同步参数

    def step(self):
        if self.iteration == 0:
            for alg, objID in self.learn_tasks.completed(num=self.num):
                grads = ray.get(objID)
                self.learn_tasks.add(alg, alg.cal_grad.remote())  # 将完成了的learner重新算梯度
            self.iteration += 1
        else:
            for alg, objID in self.learn_tasks.completed():
                grads = ray.get(objID)
                # if self.iteration > 2 * self.num:
                P_sub = ray.get(alg.state_dict.remote())
                if np.sum(P_sub - self.networks.P) != 0:
                    P = self.networks.P - P_sub
                    H = ray.get(alg.cal_hessian.remote())
                    modify = ray.get(alg.cal_modify.remote(H, P))
                    grads = grads + modify
                grad_main = self.networks.cal_grad()
                if np.abs(np.sum(grad_main - grads))/np.abs(np.sum(grad_main)) > 0.1:
                    H = ray.get(alg.cal_hessian.remote())
                    modify = ray.get(alg.cal_modify.remote(H, self.networks.P - P_sub))
                    print('H = ', H)
                    print('error_point = ', self.iteration)
                    print('P_sub = ', P_sub)
                    print('self.networks.P = ', self.networks.P)
                    print('P_error = ', self.networks.P - P_sub)
                    print('grad_original = ', grads - modify)
                    print('grad_expect = ', grad_main)
                    print('grad_exact = ', grads)
                    print('modify = ', modify)
                self.P_sub_recode_0.append(grads[0, 0])
                self.P_sub_recode_1.append(grads[0, 1])
                self.P_sub_recode_2.append(grads[1, 0])
                self.P_sub_recode_3.append(grads[1, 1])
                self.P_main_recode_0.append(grad_main[0, 0])
                self.P_main_recode_1.append(grad_main[0, 1])
                self.P_main_recode_2.append(grad_main[1, 0])
                self.P_main_recode_3.append(grad_main[1, 1])
                self.networks.update_P(grad_main)
                weights = self.networks.P
                alg.load_state_dict.remote(weights)  # 更新learner参数
                self.learn_tasks.add(alg, alg.cal_grad.remote())  # 将完成了的learner重新算梯度
                self.iteration += 1
                # print('iteration = ', self.iteration)


    def train(self):
        while True:
            self.step()
            if self.iteration > 1:
                if np.abs(np.sum(self.networks.P - self.P_old)) < 0.0001:
                    print('successful')
                    print('ite =', self.iteration)
                    # plt.subplot(221)
                    # plt.plot(np.abs(np.array(self.P_main_recode_0) - np.array(self.P_sub_recode_0)) / np.array(
                    #     self.P_main_recode_0))
                    # plt.ylim(-1, 1)
                    # plt.subplot(222)
                    # plt.plot(np.abs(np.array(self.P_main_recode_1) - np.array(self.P_sub_recode_1)) / np.array(
                    #     self.P_main_recode_1))
                    # plt.ylim(-1, 1)
                    # plt.subplot(223)
                    # plt.plot(np.abs(np.array(self.P_main_recode_2) - np.array(self.P_sub_recode_2)) / np.array(
                    #     self.P_main_recode_2))
                    # plt.ylim(-1, 1)
                    # plt.subplot(224)
                    # plt.plot(np.abs(np.array(self.P_main_recode_3) - np.array(self.P_sub_recode_3)) / np.array(
                    #     self.P_main_recode_3))
                    # plt.ylim(-1, 1)
                    # plt.show()
                    break

            self.P_old = copy.deepcopy(self.networks.P)
            if self.iteration > 1000:
                print('warning')
                break

        # self.iteration = 0