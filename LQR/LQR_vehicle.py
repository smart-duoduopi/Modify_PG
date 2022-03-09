import numpy as np
import copy
import matplotlib.pyplot as plt
from sample_trajectory.path_tracking_env import VehicleDynamics
from sample_trajectory.ini_state import Init_state

def cal_u(R, B, P, A, x):
    mid = - np.linalg.inv(R + np.matmul(np.matmul(B.transpose(), P), B)) * np.matmul(np.matmul(B.transpose(), P), A)
    u = mid[0, 0] * x[:, 0] + mid[0, 1] * x[:, 1] + mid[0, 2] * x[:, 2] + mid[0, 3] * x[:, 3]
    return u

def eva_u(R, B, P, A, x):
    mid = - np.linalg.inv(R + np.matmul(np.matmul(B.transpose(), P), B)) * np.matmul(np.matmul(B.transpose(), P), A)
    u = mid[0, 0] * x[0] + mid[0, 1] * x[1] + mid[0, 2] * x[2] + mid[0, 3] * x[3]
    return u

def state_trans(A, B, x, u):
    x_next = [A[0][0] * x[:, 0] + A[0][1] * x[:, 1] + A[0][2] * x[:, 2] + A[0][3] * x[:, 3] + B[0][0] * u,
                A[1][0] * x[:, 0] + A[1][1] * x[:, 1] + A[1][2] * x[:, 2] + A[1][3] * x[:, 3] + B[1][0] * u,
                A[2][0] * x[:, 0] + A[2][1] * x[:, 1] + A[2][2] * x[:, 2] + A[2][3] * x[:, 3] + B[2][0] * u,
                A[3][0] * x[:, 0] + A[3][1] * x[:, 1] + A[3][2] * x[:, 2] + A[3][3] * x[:, 3] + B[3][0] * u]
    state_new = np.stack(x_next, 1)
    return state_new

def eva_trans(A, B, x, u):
    x_next = [A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2] + A[0][3] * x[3] + B[0][0] * u,
                A[1][0] * x[0] + A[1][1] * x[1] + A[1][2] * x[2] + A[1][3] * x[3] + B[1][0] * u,
                A[2][0] * x[0] + A[2][1] * x[1] + A[2][2] * x[2] + A[2][3] * x[3] + B[2][0] * u,
                A[3][0] * x[0] + A[3][1] * x[1] + A[3][2] * x[2] + A[3][3] * x[3] + B[3][0] * u]
    # state_new = np.stack(x_next, 1)
    return x_next

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
        p_grad.append(np.multiply((np.matmul(np.matmul(x[i, :].transpose(), Q), x[i, :]) + u[i].transpose() * R[0, 0] * u[i] + np.matmul(
            np.matmul(x_next[i, :].transpose(), P), x_next[i, :]) - np.matmul(np.matmul(x[i, :].transpose(), P), x[i, :])), error_matrix))
    return p_grad


if __name__ == '__main__':
    A = VehicleDynamics().A
    B = VehicleDynamics().B
    Q = VehicleDynamics().Q
    R = VehicleDynamics().R
    alpha = 0.001
    P_init = np.zeros((4, 4))
    P_old = copy.deepcopy(P_init)
    P_new = copy.deepcopy(P_init)
    num_state = 500
    num = 0
    while True:
        ini = Init_state(num_state)
        train_set = ini.reset()
        x = train_set
        u = cal_u(R, B, P_new, A, x)
        x_next = state_trans(A, B, x, u)
        grad = cal_grad(P_old, x, u, x_next, Q, R)
        grad_mean = np.mean(grad, axis=0)
        print('grad = ', grad)
        print('grad_mean = ', grad_mean)
        exit()
        P_new = update_P(P_old, grad_mean)
        if np.abs(np.sum(P_new - P_old)) < 0.00000001:
            print('success')
            print('evaulation')
            y = []
            e_x = x[0]
            y.append(e_x[2])
            for i in range(100):
                e_u = eva_u(R, B, P_new, A, e_x)
                e_x = eva_trans(A, B, e_x, e_u)
                y.append(e_x[2])
            plt.plot(y)
            plt.show()
            break
        P_old = copy.deepcopy(P_new)

        if num % 500 == 0:
            print('P_new = ', P_new)
        num = num + 1
        print('num = ', num)
    print('P_terminal = ', P_new)

