import numpy as np
from sync_server.linear_state_function_regulator_env import VehicleDynamics

class System_define():
    def __init__(self,):
        Vehicle = VehicleDynamics()
        # A = np.zeros((2, 2))
        # A[1, 0] = 0.01
        # A[0, 1] = 1
        # B = np.zeros((2, 1))
        # B[1, 0] = 1
        # Q = np.zeros((2, 2))
        # Q[0, 0] = 1
        # Q[1, 1] = 100
        # R = 1
        self.A = Vehicle.A
        self.B = Vehicle.B
        self.Q = Vehicle.Q
        self.R = Vehicle.R
        self.P = np.zeros_like(self.A)
        self.alpha = 0.001
        self.x = None

    def reset_x(self, x):
        self.x = x
        self.cal_u()
        self.state_trans()

    def value_function(self,):
        self.value = np.matmul(np.matmul(self.x.transpose(), self.P), self.x)

    def cal_u(self):
        self.u = - np.linalg.inv(self.R + np.matmul(np.matmul(self.B.transpose(), self.P), self.B)) * np.matmul(
            np.matmul(np.matmul(self.B.transpose(), self.P), self.A), self.x)

    def state_trans(self,):
        self.x_next = np.matmul(self.A, self.x) + np.matmul(self.B, self.u)

    def update_P(self, grads):
        self.P = self.P - self.alpha * grads

    def P_init(self):
        self.P = np.zeros_like(self.A)

    def cal_grad(self,):
        P_grad = (np.matmul(np.matmul(self.x.transpose(), self.Q), self.x) + self.u.transpose() * self.R * self.u +
                    np.matmul(np.matmul(self.x_next.transpose(), self.P), self.x_next) -
                    np.matmul(np.matmul(self.x.transpose(), self.P), self.x)) * (np.matmul(self.x_next, self.x_next.transpose()) - np.matmul(self.x, self.x.transpose()))
        if P_grad.any == None:
            print('non_grad')
            return np.zeros_like(self.P)

        return P_grad

    def cal_vector(self, x):
        vert = x.shape[0]
        horizon = x.shape[1]
        x_vert = np.zeros((vert * horizon, 1))
        for i in range(horizon):
            x_vert[i*vert:(i+1) * vert, 0] = x[:, i]
        return x_vert

    def cal_hessian(self, ):
        x_sqrt = np.matmul(self.x, self.x.transpose())
        x_sqrt_vertor = self.cal_vector(x_sqrt)
        x_next_sqrt = np.matmul(self.x_next, self.x_next.transpose())
        x_next_sqrt_vertor = self.cal_vector(x_next_sqrt)
        error_sqrt = x_next_sqrt - x_sqrt
        error_sqrt_vertor = self.cal_vector(error_sqrt)
        H = np.matmul(x_next_sqrt_vertor, error_sqrt_vertor.transpose()) -\
            np.matmul(x_sqrt_vertor, error_sqrt_vertor.transpose())
        # H = np.matmul(x_sqrt_vertor, x_sqrt_vertor.transpose())
        return H

    def cal_modify(self, H, P):
        vec_P = np.zeros((4, 1))
        vec_P[0, 0] = P[0, 0]
        vec_P[1, 0] = P[1, 0]
        vec_P[2, 0] = P[0, 1]
        vec_P[3, 0] = P[1, 1]
        delta = np.matmul(H, vec_P)
        modify = np.zeros((2, 2))
        modify[0, 0] = delta[0, 0]
        modify[1, 0] = delta[1, 0]
        modify[0, 1] = delta[2, 0]
        modify[1, 1] = delta[3, 0]

        return modify

    def load_state_dict(self, P):
        self.P = P

    def state_dict(self):
        return self.P

    def reset(self,):
        num_agent = 1
        init_y = np.random.normal(0, 0.5, (num_agent,)).astype(np.float32)
        init_v_x = np.random.uniform(20, 20, (num_agent,)).astype(np.float32)
        beta = np.random.normal(0, 0.15, (num_agent,)).astype(np.float32)
        init_v_y = init_v_x * np.tan(beta)
        init_r = np.random.normal(0, 0.3, (num_agent,)).astype(np.float32)
        init_phi = np.random.normal(0, 0.15, (num_agent,)).astype(np.float32)
        x_init = np.array([init_v_y, init_r, init_y, init_phi])

        return x_init

if __name__=="__main__":
    A = System_define()
    x = np.random.rand(4, 3)
    x_vert = A.cal_vector(x)
    print('x = ', x)
    print('x_vert = ', x_vert)

