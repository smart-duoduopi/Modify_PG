import numpy as np
import copy
from sample_trajectory.ini_state import Init_state


def cal_u(R, B, P, A, x):
    mid = - np.linalg.inv(R + np.matmul(np.matmul(B.transpose(), P), B)) * np.matmul(np.matmul(B.transpose(), P), A)
    u = mid[0, 0] * x[:, 0] + mid[0, 1] * x[:, 1]
    return u

def state_trans(A, B, x, u):
    x_next = [A[0][0] * x[:, 0] + A[0][1] * x[:, 1] + B[0][0] * u,
                A[1][0] * x[:, 0] + A[1][1] * x[:, 1] + B[1][0] * u]
    state_new = np.stack(x_next, 1)
    return state_new

def update_P(P, alpha, grad):
    P_next = P - alpha * grad
    return P_next

def cal_grad(P, x, u, x_next, Q, R):
    n = np.shape(x)[0]
    p_grad = []
    for i in range(n):
        xx = [[x[i, :][0] * x[i, :][0], x[i, :][0] * x[i, :][1]],
              [x[i, :][1] * x[i, :][0], x[i, :][1] * x[i, :][1]]]
        xx_next = [
            [x_next[i, :][0] * x_next[i, :][0], x_next[i, :][0] * x_next[i, :][1]],
            [x_next[i, :][1] * x_next[i, :][0], x_next[i, :][1] * x_next[i, :][1]]]

        error_matrix = [[xx_next[0][0] - xx[0][0], xx_next[0][1] - xx[0][1]],
                        [xx_next[1][0] - xx[1][0], xx_next[1][1] - xx[1][1]]]

        p_grad.append(np.multiply(
            (np.matmul(np.matmul(x[i, :].transpose(), Q), x[i, :]) + u[i].transpose() * R[0, 0] * u[i] + np.matmul(
                np.matmul(x_next[i, :].transpose(), P), x_next[i, :]) - np.matmul(np.matmul(x[i, :].transpose(), P),
                                                                                  x[i, :])), error_matrix))
    return p_grad

if __name__ == '__main__':
    A = np.zeros((2, 2))
    A[1, 0] = 0.01
    A[0, 1] = 1
    B = np.zeros((2, 1))
    B[1, 0] = 1
    Q = np.zeros((2, 2))
    Q[0, 0] = 1
    Q[1, 1] = 100
    R = np.ones((1, 1))
    alpha = 0.0001
    P_init = np.zeros((2, 2))
    P_old = copy.deepcopy(P_init)
    P_new = copy.deepcopy(P_init)
    num_state = 50
    num = 0
    while True:
        ini = Init_state(num_state)
        train_set = ini.reset_two()
        x = train_set
        u = cal_u(R, B, P_new, A, x)
        x_next = state_trans(A, B, x, u)
        grad = cal_grad(P_old, x, u, x_next, Q, R)
        grad_mean = np.mean(grad, axis=0)
        P_new = update_P(P_old, alpha, grad_mean)
        if np.abs(np.sum(P_new - P_old)) < 0.00000001:
            print('success')
            break
        P_old = copy.deepcopy(P_new)
        num = num + 1
        print('num = ', num)
    print('P_terminal = ', P_new)

