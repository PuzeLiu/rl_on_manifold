import numpy as np
from atacom.environments.circular_motion.circle_base import CircularMotion

from mushroom_rl.core import Core, Agent
from mushroom_rl.utils.dataset import compute_J


class CircleEnvTerminated(CircularMotion):
    """
    CircularMotion with termination when constraint > tolerance
    """

    def __init__(self, time_step=0.01, horizon=500, gamma=0.99, random_init=False, tol=0.1):
        super().__init__(time_step=time_step, horizon=horizon, gamma=gamma, random_init=random_init)
        self._tol = tol

    def step(self, action):
        state, reward, absorbing, info = super().step(action)

        absorbing = absorbing or self._terminate()
        if absorbing:
            reward = -100
        return state, reward, absorbing, info

    def _terminate(self):
        if np.any(self.c > self._tol):
            return True
        else:
            return False

    def render(self):
        offset = np.array([1.25, 1.25])
        pos = self._state[:2] + offset

        act_b = self._action[:2]
        self._viewer.force_arrow(center=pos, direction=act_b,
                                 force=np.linalg.norm(act_b),
                                 max_force=10, width=5,
                                 max_length=0.3, color=(0, 255, 255))
        super().render()


def env_test():
    env = CircleEnvTerminated(tol=0.1)

    class DummyAgent(Agent):
        def __init__(self, mdp_info):
            self.mdp_info = mdp_info

        def fit(self, dataset):
            pass

        def episode_start(self):
            pass

        def draw_action(self, state):
            return np.random.randn(self.mdp_info.action_space.shape[0]) * 1

    agent = DummyAgent(env.info)

    core = Core(agent, env)

    dataset = core.evaluate(n_steps=1000, render=True)

    J = np.mean(compute_J(dataset, core.mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    c_avg, c_max, c_dq_max = env.get_constraints_logs()
    print("J: {}, R:{}, c_avg:{}, c_max:{}, c_dq_max:{}".format(J, R, c_avg, c_max, c_dq_max))


if __name__ == '__main__':
    env_test()
