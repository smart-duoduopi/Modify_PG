#   Copyright (c) 2021 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Jiaxin Gao

import time
import numpy as np
import ray
import matplotlib.pyplot as plt
from sync_server.model_define import System_define
from sync_server.async_trainer import async_trainer
from sync_server.async_trainer_modify import async_trainer_modify
from sync_server.serial_trainer import serial_trainer


if __name__=='__main__':
    ######## parallel
    # num = 2
    # ray.init()
    # algs = [ray.remote(num_cpus=1)(System_define).remote()
    #        for _ in range(num)]
    # main_train = async_trainer_modify(algs, num)
    ######## serial
    algs = System_define()
    main_train = serial_trainer(algs)
    #############
    for i in range(2000):
        print('i = ', i)
        # x = np.array([[-1.283], [-1.5], [2.8], [1.559]])
        x = 0.5 * np.ones((4, 1)) + np.random.rand(4, 1)
        # x = np.ones((2, 1)) + np.random.rand(2, 1)
        # x = algs.reset()
        # print('x = ', x)
        # exit()
        main_train.networks.reset_x(x)
        ######## parallel
        # for alg in algs:
        #     alg.reset_x.remote(x)
        ######## serial
        algs.reset_x(x)
        #############
        if i == 0:
            main_train._set_algs()
        # else:
        #     main_train.sync()

        main_train.train()

        main_train.iteration = 0
        weights = main_train.networks.state_dict()

        ######## parallel
        # for alg in algs:
        #     alg.load_state_dict.remote(weights)
        ######## serial
        algs.load_state_dict(weights)
        #############
        if np.abs(main_train.networks.P[0, 0] - 1) < 0.001 and np.abs(main_train.networks.P[0, 1] - 0) < 0.001 and \
                np.abs(main_train.networks.P[1, 0] - 0) < 0.001 and np.abs(main_train.networks.P[1, 1] - 101) < 0.001:
            break

    print(main_train.networks.P)
    # plt.subplot(221)
    # plt.plot(np.abs(np.array(main_train.P_main_recode_0) - np.array(main_train.P_sub_recode_0)) / np.array(main_train.P_main_recode_0))
    # plt.ylim(-1, 1)
    # plt.subplot(222)
    # plt.plot(np.abs(np.array(main_train.P_main_recode_1) - np.array(main_train.P_sub_recode_1)) / np.array(main_train.P_main_recode_1))
    # plt.ylim(-1, 1)
    # plt.subplot(223)
    # plt.plot(np.abs(np.array(main_train.P_main_recode_2) - np.array(main_train.P_sub_recode_2)) / np.array(main_train.P_main_recode_2))
    # plt.ylim(-1, 1)
    # plt.subplot(224)
    # plt.plot(np.abs(np.array(main_train.P_main_recode_3) - np.array(main_train.P_sub_recode_3)) / np.array(main_train.P_main_recode_3))
    # plt.ylim(-1, 1)
    # plt.show()


