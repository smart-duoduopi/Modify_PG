import numpy as np
import copy
import matplotlib.pyplot as plt

def value_function(P, x):
    V = np.matmul(np.matmul(x.transpose(), P), x)
    return V

def cal_u(R, B, P, A, x):
    u = - np.linalg.inv(R + np.matmul(np.matmul(B.transpose(), P), B)) * np.matmul(np.matmul(np.matmul(B.transpose(), P), A), x)
    return u

def state_trans(A, B, x, u):
    x_next = np.matmul(A, x) + np.matmul(B, u)
    return x_next

def update_P(P, alpha, x, u, x_next, Q, R):
    P_next = P + alpha * (np.matmul(np.matmul(x.transpose(), Q), x) + u.transpose() * R * u + np.matmul(np.matmul(x_next.transpose(), P), x_next) - np.matmul(np.matmul(x.transpose(), P), x)) * np.matmul(x, x.transpose())
    return P_next

def cal_grad(P, x, u, x_next, Q, R):
    P_grad = (np.matmul(np.matmul(x.transpose(), Q), x) + u.transpose() * R * u + np.matmul(
        np.matmul(x_next.transpose(), P), x_next) - np.matmul(np.matmul(x.transpose(), P), x)) * \
             (np.matmul(x_next, x_next.transpose()) - np.matmul(x, x.transpose()))
    return P_grad


if __name__ == '__main__':
    A = np.zeros((2, 2))
    A[1, 0] = 0.01
    A[0, 1] = 1
    B = np.zeros((2, 1))
    B[1, 0] = 1
    Q = np.zeros((2, 2))
    Q[0, 0] = 1
    Q[1, 1] = 100
    R = 1
    alpha = 0.0001
    P_init = np.zeros((2, 2))
    P_old = copy.deepcopy(P_init)
    P_new = copy.deepcopy(P_init)
    P_ter = copy.deepcopy(P_init)
    for _ in range(2000):
        print('ite = ', _)
        if _ > 0:
            if np.abs(np.sum(P_new - P_ter)) < 0.0000001:
                print('final_success')
                break
            P_ter = copy.deepcopy(P_new)
        x = 4 * np.random.rand(2, 1)
        u = cal_u(R, B, P_new, A, x)
        x_next = state_trans(A, B, x, u)
        num = 0
        while True:
            P_new = update_P(P_old, alpha, x, u, x_next, Q, R)
            if np.abs(np.sum(P_new - P_old)) < 0.0001:
                print('success')
                break
            P_old = copy.deepcopy(P_new)

    print('P_terminal = ', P_new)

