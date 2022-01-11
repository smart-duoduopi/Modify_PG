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
    P_grad = - (np.matmul(np.matmul(x.transpose(), Q), x) + u.transpose() * R * u + np.matmul(
        np.matmul(x_next.transpose(), P), x_next) - np.matmul(np.matmul(x.transpose(), P), x)) * np.matmul(x, x.transpose())
    return P_grad

def cal_hessian(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    H = np.zeros((4, 4))
    H[0, 0] = x1 * x1 * x1 * x1
    H[0, 1] = x1 * x2 * x1 * x1
    H[0, 2] = x1 * x2 * x1 * x1
    H[0, 3] = x2 * x2 * x1 * x1
    H[1, 0] = x1 * x2 * x1 * x1
    H[1, 1] = x1 * x2 * x1 * x2
    H[1, 2] = x1 * x2 * x1 * x2
    H[1, 3] = x2 * x2 * x1 * x2
    H[2, 0] = x1 * x1 * x1 * x2
    H[2, 1] = x1 * x2 * x1 * x2
    H[2, 2] = x1 * x2 * x1 * x2
    H[2, 3] = x2 * x2 * x1 * x2
    H[3, 0] = x1 * x1 * x2 * x2
    H[3, 1] = x1 * x2 * x2 * x2
    H[3, 2] = x1 * x2 * x2 * x2
    H[3, 3] = x2 * x2 * x2 * x2
    # H = np.matmul(np.matmul(x, x.transpose()), np.matmul(x, x.transpose()))
    return H


def cal_modify(H, P):
    vec_P = np.zeros((4, 1))
    vec_P[0, 0] = P[0, 0]
    vec_P[1, 0] = P[1, 0]
    vec_P[2, 0] = P[0, 1]
    vec_P[3, 0] = P[1, 1]
    # print('H = ', H)
    # print('vec_P = ', vec_P)
    # print('P =', P)
    delta = np.matmul(H, vec_P)
    # print(delta)
    modify = np.zeros((2, 2))
    modify[0, 0] = delta[0, 0]
    modify[1, 0] = delta[1, 0]
    modify[0, 1] = delta[2, 0]
    modify[1, 1] = delta[3, 0]
    # print(modify)
    # exit()

    return modify


if __name__ == '__main__' :
    # B = np.zeros((4, 1))
    # B[0, 0] = 0.0298
    # B[1, 0] = 0.0283
    # B[2, 0] = 3.569*1e-4
    # B[3, 0] = 2.891*1e-4
    # A = np.zeros((4, 4))
    # A[0, 0] = 0.9065
    # A[1, 0] = 0.0542
    # A[2, 0] = 0.0192
    # A[3, 0] = 5.6966*1e-4
    # A[0, 1] = -0.2805
    # A[1, 1] = 0.8138
    # A[2, 1] = 7.989*1e-4
    # A[3, 1] = 0.0181
    # A[0, 2] = 0
    # A[1, 2] = 0
    # A[2, 2] = 1
    # A[3, 2] = 0
    # A[0, 3] = 0
    # A[1, 3] = 0
    # A[2, 3] = 0.04
    # A[3, 3] = 1
    # R = 1
    # Q = np.zeros((4, 4))
    # Q[2, 2] = 36
    # Q[3, 3] = 3
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
    P_m = copy.deepcopy(P_init)
    x_m = np.random.rand(2, 1)
    P_old_modify_recode_0 = []
    P_old_modify_recode_1 = []
    P_old_modify_recode_2 = []
    P_old_modify_recode_3 = []
    P_new_grad_recode_0 = []
    P_new_grad_recode_1 = []
    P_new_grad_recode_2 = []
    P_new_grad_recode_3 = []
    for _ in range(2000):
        print('ite = ', _)
        x = 4 * np.random.rand(2, 1)
        u = cal_u(R, B, P_new, A, x)
        x_next = state_trans(A, B, x, u)
        num = 0
        while True:
            P_new = update_P(P_old, alpha, x, u, x_next, Q, R)
            num = num + 1
            if num > 1:
                P_old_grad = cal_grad(P_m, x, u, x_next, Q, R)
                H = cal_hessian(x)
                modify = cal_modify(H, P_new - P_m)
                P_old_modify = P_old_grad + modify
                P_new_grad = cal_grad(P_new, x, u, x_next, Q, R)
                P_old_modify_recode_0.append(P_old_modify[0, 0])
                P_old_modify_recode_1.append(P_old_modify[0, 1])
                P_old_modify_recode_2.append(P_old_modify[1, 0])
                P_old_modify_recode_3.append(P_old_modify[1, 1])
                P_new_grad_recode_0.append(P_new_grad[0, 0])
                P_new_grad_recode_1.append(P_new_grad[0, 1])
                P_new_grad_recode_2.append(P_new_grad[1, 0])
                P_new_grad_recode_3.append(P_new_grad[1, 1])
                P_m = copy.deepcopy(P_new)
            if np.abs(np.sum(P_new - P_old)) < 0.0001:
                break
            P_old = copy.deepcopy(P_new)


    plt.subplot(221)
    plt.plot((np.array(P_new_grad_recode_0) - np.array(P_old_modify_recode_0))/np.array(P_new_grad_recode_0))
    plt.ylim(-10, 10)
    plt.subplot(222)
    plt.plot((np.array(P_new_grad_recode_1) - np.array(P_old_modify_recode_1))/np.array(P_new_grad_recode_1))
    plt.ylim(-10, 10)
    plt.subplot(223)
    plt.plot((np.array(P_new_grad_recode_2) - np.array(P_old_modify_recode_2))/np.array(P_new_grad_recode_2))
    plt.ylim(-10, 10)
    plt.subplot(224)
    plt.plot((np.array(P_new_grad_recode_3) - np.array(P_old_modify_recode_3))/np.array(P_new_grad_recode_3))
    plt.ylim(-10, 10)
    plt.show()

    print('P_terminal = ', P_new)

