import numpy as np
from atacom.environments.circular_motion.circle_base import CircularMotion
from atacom.atacom import AtacomEnvWrapper
from atacom.constraints import ViabilityConstraint, ConstraintsSet

class CircleEnvAtacom(AtacomEnvWrapper):
    def __init__(self, horizon=500, gamma=0.99, random_init=False, Kc=100, time_step=0.01):
        base_env = CircularMotion(random_init=random_init, horizon=horizon, gamma=gamma)
        circle_constr = ViabilityConstraint(2, 1, fun=self.circle_fun, J=self.circle_J, b=self.circle_b, K=0.1)

        height_constr = ViabilityConstraint(2, 1, fun=self.height_fun, J=self.height_J, b=self.height_b, K=2)

        f = ConstraintsSet(2)
        f.add_constraint(circle_constr)
        g = ConstraintsSet(2)
        g.add_constraint(height_constr)

        super().__init__(base_env=base_env, dim_q=2, f=f, g=g, Kc=Kc, acc_max=10, vel_max=1, Kq=20, time_step=time_step)

    def _get_q(self, state):
        return state[:2]

    def _get_dq(self, state):
        return state[2:4]

    def acc_to_ctrl_action(self, ddq):
        return ddq / self.acc_max

    def render(self):
        offset = np.array([1.25, 1.25])
        pos = self.state[:2] + offset

        act_a = self._act_a[:2]
        act_b = self._act_b[:2]

        self.env._viewer.force_arrow(center=pos, direction=act_a,
                                     force=np.linalg.norm(act_a),
                                     max_force=3, width=5,
                                     max_length=0.3, color=(255, 165, 0))

        self.env._viewer.force_arrow(center=pos, direction=act_b,
                                     force=np.linalg.norm(act_b),
                                     max_force=10, width=5,
                                     max_length=0.3, color=(0, 255, 255))
        super().render()

    @staticmethod
    def circle_fun(q):
        return np.array([q[0] ** 2 + q[1] ** 2 - 1])

    @staticmethod
    def circle_J(q):
        return np.array([[2 * q[0], 2 * q[1]]])

    @staticmethod
    def circle_b(q, dq):
        return np.array([[2 * dq[0], 2 * dq[1]]]) @ dq

    @staticmethod
    def height_fun(q):
        return np.array([-q[1] - 0.5])

    @staticmethod
    def height_J(q):
        return np.array([[0, -1]])

    @staticmethod
    def height_b(q, dq):
        return np.array([0])

    @staticmethod
    def vel_fun(q, dq):
        return np.array([dq[0] ** 2 + dq[1] ** 2 - 1])

    @staticmethod
    def vel_A(q, dq):
        return np.array([[2 * dq[0], 2 * dq[1]]])

    @staticmethod
    def vel_b(q, dq):
        return np.array([0.])
