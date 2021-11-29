import numpy as np

from mushroom_rl.utils import spaces
from atacom.utils import pinv_null


class ErrorCorrectionEnvWrapper:
    """
    Environment wrapper of the Error Correction Method

    """
    def __init__(self, base_env, dim_q, vel_max, acc_max, f=None, g=None, Kc=100., Kq=10., time_step=0.01):
        """
        Constructor
        Args:
            base_env (mushroomrl.Core.Environment): The base environment inherited from
            dim_q (int): [int] dimension of the directly controllable variable
            vel_max (array, float): the maximum velocity of the directly controllable variable
            acc_max (array, float): the maximum acceleration of the directly controllable variable
            f (ViabilityConstraint, ConstraintsSet): the equality constraint f(q) = 0
            g (ViabilityConstraint, ConstraintsSet): the inequality constraint g(q) = 0
            Kc (array, float): the scaling factor for error correction
            Ka (array, float): the scaling factor for the viability acceleration bound
            time_step (float): the step size for time discretization
        """
        self.env = base_env
        self.dims = {'q': dim_q, 'f': 0, 'g': 0}
        self.f = f
        self.g = g
        self.time_step = time_step
        self._logger = None

        if self.f is not None:
            assert self.dims['q'] == self.f.dim_q, "Input dimension is different in f"
            self.dims['f'] = self.f.dim_out
        if self.g is not None:
            assert self.dims['q'] == self.g.dim_q, "Input dimension is different in g"
            self.dims['g'] = self.g.dim_out
            self.s = np.zeros(self.dims['g'])

        self.dims['c'] = self.dims['f'] + self.dims['g']

        if np.isscalar(Kc):
            self.K_c = np.ones(self.dims['c']) * Kc
        else:
            self.K_c = Kc

        self.q = np.zeros(self.dims['q'])
        self.dq = np.zeros(self.dims['q'])

        self._mdp_info = self.env.info.copy()
        self._mdp_info.action_space = spaces.Box(low=-np.ones(self.dims['q']), high=np.ones(self.dims['q']))

        if np.isscalar(vel_max):
            self.vel_max = np.ones(self.dims['q']) * vel_max
        else:
            self.vel_max = vel_max
            assert np.shape(self.vel_max)[0] == self.dims['q']

        if np.isscalar(acc_max):
            self.acc_max = np.ones(self.dims['q']) * acc_max
        else:
            self.acc_max = acc_max
            assert np.shape(self.acc_max)[0] == self.dims['q']

        if np.isscalar(Kq):
            self.K_q = np.ones(self.dims['q']) * Kq
        else:
            self.K_q = Kq
            assert np.shape(self.K_q)[0] == self.dims['q']

        self.state = self.env.reset()
        self._act_a = None
        self._act_b = None
        self._act_err = None

        self.env.step_action_function = self.step_action_function

    def acc_to_ctrl_action(self, ddq):
        raise NotImplementedError

    def _get_q(self, state):
        raise NotImplementedError

    def _get_dq(self, state):
        raise NotImplementedError

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self, state=None):
        self.state = self.env.reset(state)
        self.q = self._get_q(self.state)
        self.dq = self._get_dq(self.state)
        self._compute_slack_variables()
        return self.state

    def render(self):
        self.env.render()

    def stop(self):
        self.env.stop()

    def step(self, action):
        alpha = np.clip(action, self.info.action_space.low, self.info.action_space.high)
        alpha = alpha * self.acc_max

        self.state, reward, absorb, info = self.env.step(alpha)
        return self.state.copy(), reward, absorb, info

    def acc_truncation(self, dq, ddq):
        acc_u = np.maximum(np.minimum(self.acc_max, -self.K_q * (dq - self.vel_max)), -self.acc_max)
        acc_l = np.minimum(np.maximum(-self.acc_max, -self.K_q * (dq + self.vel_max)), self.acc_max)
        ddq = np.clip(ddq, acc_l, acc_u)
        return ddq

    def step_action_function(self, sim_state, alpha):
        self.state = self.env._create_observation(sim_state)
        self.q = self._get_q(self.state)
        self.dq = self._get_dq(self.state)

        Jc, psi = self._construct_Jc_psi(self.q, self.s, self.dq)
        Jc_inv, Nc = pinv_null(Jc)

        self._act_a = np.zeros(self.dims['q'] + self.dims['g'])
        self._act_b = np.concatenate([alpha, np.zeros(self.dims['g'])])
        self._act_err = self._compute_error_correction(self.q, self.dq, self.s, Jc_inv)
        ddq_ds = self._act_a + self._act_b + self._act_err

        self.s += ddq_ds[self.dims['q']:(self.dims['q'] + self.dims['g'])] * self.time_step

        ddq = self.acc_truncation(self.dq, ddq_ds[:self.dims['q']])
        ctrl_action = self.acc_to_ctrl_action(ddq)
        return ctrl_action

    @property
    def info(self):
        return self._mdp_info

    def _compute_slack_variables(self):
        self.s = None
        if self.dims['g'] > 0:
            s_2 = np.maximum(-2 * self.g.fun(self.q, self.dq, origin_constr=False), 0)
            self.s = np.sqrt(s_2)

    def _construct_Jc_psi(self, q, s, dq):
        Jc = np.zeros((self.dims['f'] + self.dims['g'], self.dims['q'] + self.dims['g']))
        psi = np.zeros(self.dims['c'])
        if self.dims['f'] > 0:
            idx_0 = 0
            idx_1 = self.dims['f']
            Jc[idx_0:idx_1, :self.dims['q']] = self.f.K_J(q)
            psi[idx_0:idx_1] = self.f.b(q, dq)
        if self.dims['g'] > 0:
            idx_0 = self.dims['f']
            idx_1 = self.dims['f'] + self.dims['g']
            Jc[idx_0:idx_1, :self.dims['q']] = self.g.K_J(q)
            Jc[idx_0:idx_1, self.dims['q']:(self.dims['q'] + self.dims['g'])] = np.diag(s)
            psi[idx_0:idx_1] = self.g.b(q, dq)
        return Jc, psi

    def _compute_error_correction(self, q, dq, s, Jc_inv, act_null=None):
        q_tmp = q.copy()
        dq_tmp = dq.copy()
        s_tmp = None

        if self.dims['g'] > 0:
            s_tmp = s.copy()

        if act_null is not None:
            q_tmp += dq_tmp * self.time_step + act_null[:self.dims['q']] * self.time_step ** 2 / 2
            dq_tmp += act_null[:self.dims['q']] * self.time_step
            if self.dims['g'] > 0:
                s_tmp += act_null[self.dims['q']:self.dims['q'] + self.dims['g']] * self.time_step

        return -Jc_inv @ (self.K_c * self._compute_c(q_tmp, dq_tmp, s_tmp, origin_constr=False))

    def _compute_c(self, q, dq, s, origin_constr=False):
        c = np.zeros(self.dims['f'] + self.dims['g'])
        if self.dims['f'] > 0:
            idx_0 = 0
            idx_1 = self.dims['f']
            c[idx_0:idx_1] = self.f.fun(q, dq, origin_constr)
        if self.dims['g'] > 0:
            idx_0 = self.dims['f']
            idx_1 = self.dims['f'] + self.dims['g']
            if origin_constr:
                c[idx_0:idx_1] = self.g.fun(q, dq, origin_constr)
            else:
                c[idx_0:idx_1] = self.g.fun(q, dq, origin_constr) + 1 / 2 * s ** 2
        return c

    def set_logger(self, logger):
        self._logger = logger

    def get_constraints_logs(self):
        return self.env.get_constraints_logs()
