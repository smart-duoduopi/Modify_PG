import numpy as np
import copy
import matplotlib.pyplot as plt
from sample_trajectory.path_tracking_env import VehicleDynamics

def cal_u(R, B, P, A, x):
    mid = np.linalg.inv(R + np.matmul(np.matmul(B.transpose(), P), B)) * np.matmul(np.matmul(B.transpose(), P), A)
    u = mid[0, 0] * x[:, 0] + mid[0, 1] * x[:, 1] + mid[0, 2] * x[:, 2] + mid[0, 3] * x[:, 3]
    return u

def state_trans(A, B, x, u):
    x_next = [A[0][0] * x[:, 0] + A[0][1] * x[:, 1] + A[0][2] * x[:, 2] + A[0][3] * x[:, 3] + B[0][0] * u,
                A[1][0] * x[:, 0] + A[1][1] * x[:, 1] + A[1][2] * x[:, 2] + A[1][3] * x[:, 3] + B[1][0] * u,
                A[2][0] * x[:, 0] + A[2][1] * x[:, 1] + A[2][2] * x[:, 2] + A[2][3] * x[:, 3] + B[2][0] * u,
                A[3][0] * x[:, 0] + A[3][1] * x[:, 1] + A[3][2] * x[:, 2] + A[3][3] * x[:, 3] + B[3][0] * u]
    state_new = np.stack(x_next, 1)
    return state_new

def update_P(P, grad):
    P_next = P - alpha * grad
    return P_next

def cal_grad(P, x, u, x_next, Q, R):
    n = np.shape(x)[0]
    p_grad = []
    for i in range(n):
        xx = [[x[i, :][0] * x[i, :][0], x[i, :][0] * x[i, :][1], x[i, :][0] * x[i, :][2], x[i, :][0] * x[i, :][3]],
            [x[i, :][1] * x[i, :][0], x[i, :][1] * x[i, :][1], x[i, :][1] * x[i, :][2], x[i, :][1] * x[i, :][3]],
            [x[i, :][2] * x[i, :][0], x[i, :][2] * x[i, :][1], x[i, :][2] * x[i, :][2], x[i, :][2] * x[i, :][3]],
            [x[i, :][3] * x[i, :][0], x[i, :][3] * x[i, :][1], x[i, :][3] * x[i, :][2], x[i, :][3] * x[i, :][3]]]
        xx_next = [[x_next[i, :][0] * x_next[i, :][0], x_next[i, :][0] * x_next[i, :][1], x_next[i, :][0] * x_next[i, :][2], x_next[i, :][0] * x_next[i, :][3]],
            [x_next[i, :][1] * x_next[i, :][0], x_next[i, :][1] * x_next[i, :][1], x_next[i, :][1] * x_next[i, :][2], x_next[i, :][1] * x_next[i, :][3]],
            [x_next[i, :][2] * x_next[i, :][0], x_next[i, :][2] * x_next[i, :][1], x_next[i, :][2] * x_next[i, :][2], x_next[i, :][2] * x_next[i, :][3]],
            [x_next[i, :][3] * x_next[i, :][0], x_next[i, :][3] * x_next[i, :][1], x_next[i, :][3] * x_next[i, :][2], x_next[i, :][3] * x_next[i, :][3]]]

        error_matrix = [[xx_next[0][0] - xx[0][0], xx_next[0][1] - xx[0][1], xx_next[0][2] - xx[0][2],
                         xx_next[0][3] - xx[0][3]],
                        [xx_next[1][0] - xx[1][0], xx_next[1][1] - xx[1][1], xx_next[1][2] - xx[1][2],
                         xx_next[1][3] - xx[1][3]],
                        [xx_next[2][0] - xx[2][0], xx_next[2][1] - xx[2][1], xx_next[2][2] - xx[2][2],
                         xx_next[2][3] - xx[2][3]],
                        [xx_next[3][0] - xx[3][0], xx_next[3][1] - xx[3][1], xx_next[3][2] - xx[3][2],
                         xx_next[3][3] - xx[3][3]]]
        # print('x = ', x[i, :])
        # print('x_next = ', x_next[i, :])
        # print('xx = ', xx)
        # print('xx_next = ', xx_next)
        # print('error = ', error_matrix)
        # print('u = ', u[i])
        # print('P = ', P)
        # print(np.multiply((np.matmul(np.matmul(x[i, :].transpose(), Q), x[i, :]) + u[i].transpose() * R[0, 0] * u[i] + np.matmul(
        #     np.matmul(x_next[i, :].transpose(), P), x_next[i, :]) - np.matmul(np.matmul(x[i, :].transpose(), P), x[i, :])), error_matrix))
        # exit()
        p_grad.append(np.multiply((np.matmul(np.matmul(x[i, :].transpose(), Q), x[i, :]) + u[i].transpose() * R[0, 0] * u[i] + np.matmul(
            np.matmul(x_next[i, :].transpose(), P), x_next[i, :]) - np.matmul(np.matmul(x[i, :].transpose(), P), x[i, :])), error_matrix))
    return p_grad


if __name__ == '__main__':
    A = VehicleDynamics().A
    B = VehicleDynamics().B
    Q = VehicleDynamics().Q
    R = VehicleDynamics().R
    alpha = 0.01
    P_init = np.ones((4, 4))
    P_old = copy.deepcopy(P_init)
    P_new = copy.deepcopy(P_init)
    train_set = np.load('LQR_vehicle.npy')
    # x = train_set[0:500, 0:4]
    # np.save('LQR_vehicle.npy', x)
    # exit()
    x = train_set
    for _ in range(1000):
        # P_ter = copy.deepcopy(P_new)
        u = cal_u(R, B, P_new, A, x)
        x_next = state_trans(A, B, x, u)
        # 从x和u到x_next的计算应该是正确的。
        inner = 0
        while True:
            # print('inner = ', inner)
            grad = cal_grad(P_old, x, u, x_next, Q, R)
            grad_mean = np.mean(grad, axis=0)
            # print('grad = ', grad)
            # print('grad_mean = ', grad_mean)
            grad_mean = np.clip(grad_mean, a_min=-100, a_max=100)
            # print('grad_mean = ', grad_mean)
            P_new = update_P(P_old, grad_mean)
            # print('P_old = ', P_old)
            # print('P_new = ', P_new)
            # exit()
            if np.abs(np.sum(P_new - P_old)) < 0.0001:
                print('success')
                break
            if inner > 10000:
                # P_new = P_ter
                print('over_size')
                break
            P_old = copy.deepcopy(P_new)
            inner = inner + 1
        print('P_old = ', P_old)
        # exit()
    print('P_terminal = ', P_new)

