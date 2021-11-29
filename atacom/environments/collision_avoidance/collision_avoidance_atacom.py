from mushroom_rl.core import Core, Agent
from mushroom_rl.utils.spaces import *
from mushroom_rl.utils.dataset import compute_J
from atacom.environments.collision_avoidance.collision_avoidance_base import PointGoalReach
from atacom.utils.null_space_coordinate import pinv_null, rref


class PointReachAtacom(PointGoalReach):
    def __init__(self, time_step=0.01, horizon=1000, gamma=0.99, n_objects=4, random_walk=False):
        super().__init__(time_step=time_step, horizon=horizon, gamma=gamma, n_objects=n_objects,
                         random_walk=random_walk)
        self.s = np.zeros(self.n_objects)
        self.Kc = np.diag(np.ones(self.n_objects) * 100)
        self.K = np.ones(self.n_objects) * 0.5

        self.constr_logs = list()

    def reset(self, state=None):
        super(PointReachAtacom, self).reset(state)
        self.q = self.get_q(self._state)
        self.dq = self.get_dq(self._state)
        self.p = self.get_p(self._state)
        self.dp = self.get_dp(self._state)

        self.s = np.sqrt(np.maximum(- 2 * self.get_c(self.q, self.p), 0.))
        # self.s = np.maximum(- self.get_c(self.q, self.p), 0.)
        return self._state

    def step(self, action):
        self.q = self.get_q(self._state)
        self.dq = self.get_dq(self._state)
        self.p = self.get_p(self._state)
        self.dp = self.get_dp(self._state)

        Jc_q = self.get_Jc_q(self.q, self.p)
        Jc_q_inv, Nc_q = pinv_null(Jc_q)

        c_origin = self.get_c(self.q, self.p)
        c_dq_i = 0.
        self.constr_logs.append([np.max(c_origin), np.max(c_dq_i)])
        c = c_origin + 1 / 2 * self.s ** 2 + self.K * (self.get_dc(self.q, self.p, self.dq, self.dp))

        Nc = rref(Nc_q, row_vectors=False)

        psi = self.get_psi(self.q, self.p, self.dq, self.dp)
        action = - Jc_q_inv @ (psi + self.Kc @ c) + Nc @ action
        self.s += action[2:] * self.time_step
        return super().step(action[:2])

    @staticmethod
    def get_q(state):
        return state[:2]

    @staticmethod
    def get_dq(state):
        return state[2:4]

    def get_p(self, state):
        p = np.zeros(2 * self.n_objects)
        for i in range(self.n_objects):
            idx = 4 * (i + 1)
            p[2 * i: 2 * i + 2] = state[idx: idx + 2]
        return p

    def get_dp(self, state):
        dp = np.zeros(2 * self.n_objects)
        for i in range(self.n_objects):
            idx = 4 * (i + 1)
            dp[2 * i: 2 * i + 2] = state[idx + 2: idx + 4]
        return dp

    def get_c(self, q, p):
        c_out = np.zeros(self.n_objects)
        for i in range(self.n_objects):
            c_out[i] = 0.6 ** 2 - np.linalg.norm(q - p[2 * i: 2 * i + 2]) ** 2
        return c_out

    def get_dc(self, q, p, dq, dp):
        dc_out = np.zeros(self.n_objects)
        for i in range(self.n_objects):
            p_i = p[2 * i:2 * i + 2]
            dp_i = dp[2 * i:2 * i + 2]
            dc_out[i] = self.get_Jp(q, p_i) @ dp_i + self.get_Jq(q, p_i) @ dq
        return dc_out

    @staticmethod
    def get_Jq(q, p_i):
        return -2 * (q - p_i)

    @staticmethod
    def get_Jp(q, p_i):
        return 2 * (q - p_i)

    @staticmethod
    def get_Hqq(q, p_i):
        return -2 * np.eye(2)

    @staticmethod
    def get_Hqp(q, p_i):
        return 2 * np.eye(2)

    @staticmethod
    def get_Hpp(q, p_i):
        return -2 * np.eye(2)

    def get_bp(self, q, p_i, dq, dp_i):
        return (p_i @ self.get_Hpp(q, p_i) + q @ self.get_Hqp(q, p_i)) @ p_i

    def get_bq(self, q, p_i, dq, dp_i):
        return (q @ self.get_Hqq(q, p_i) + p_i @ self.get_Hqp(q, p_i)) @ q

    def get_Jc_q(self, q, p):
        Jc_q = np.zeros((self.n_objects, self.n_objects + 2))
        for i in range(self.n_objects):
            p_i = p[2 * i:2 * i + 2]
            Jc_q[i, :2] = self.get_Jq(q, p_i)
        Jc_q[:, 2:] = np.diag(self.s)
        return Jc_q

    def get_psi(self, q, p, dq, dp):
        psi = np.zeros(self.n_objects)
        for i in range(self.n_objects):
            p_i = p[2 * i:2 * i + 2]
            dp_i = dp[2 * i:2 * i + 2]
            psi[i] = self.get_Jp(q, p_i) @ dp_i + self.get_Jq(q, p_i) @ dq + \
                     self.K[i] * (self.get_bp(q, p_i, dq, dp_i) + self.get_bq(q, p_i, dq, dp_i))
        return psi

    def get_constraints_logs(self):
        constr_logs = np.array(self.constr_logs)
        c_avg = np.mean(constr_logs[:, 0])
        c_max = np.max(constr_logs[:, 0])
        c_dq_max = np.max(constr_logs[:, 1])
        self.constr_logs.clear()
        return c_avg, c_max, c_dq_max


