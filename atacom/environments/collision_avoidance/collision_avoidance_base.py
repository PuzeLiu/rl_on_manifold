from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *
from mushroom_rl.utils.viewer import Viewer


class PointGoalReach(Environment):
    def __init__(self, time_step=0.01, horizon=500, gamma=0.99, n_objects=2, random_walk=False):
        self.time_step = time_step
        self.n_objects = n_objects
        self.random_walk = random_walk

        self.state_dim = 4 * (1 + n_objects)
        observation_space = Box(low=-np.ones(self.state_dim) * 10, high=np.ones(self.state_dim) * 10)
        action_space = Box(low=-np.ones(2) * 1, high=np.ones(2) * 1)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        self._viewer = Viewer(env_width=10, env_height=10, background=(255, 255, 255))
        self.step_action_function = None
        self.action_scale = np.array([10., 10.])
        self._obj_circle_center = []
        self._obj_radius = 2
        self._time = 0.
        super().__init__(mdp_info)

    def reset(self, state=None):
        self._time = 0.
        self._state = np.zeros(self.state_dim)

        self._state[:2] = np.array([1., 1.])
        self._state[2:4] = np.array([0., 0.])

        for i in range(self.n_objects):
            obj_idx = 4 * (i + 1)
            self._state[obj_idx:obj_idx + 2] = np.random.uniform(2, 8, 2)
            self._obj_circle_center.append(self._state[obj_idx:obj_idx + 2] - np.array([self._obj_radius, 0.]))
            self._state[obj_idx + 2:obj_idx + 4] = np.zeros(2)

        return self._state

    def step(self, action):
        if self.step_action_function is not None:
            action = self.step_action_function(self._state, action)

        self._action = np.clip(action, self.info.action_space.low, self.info.action_space.high)
        self._action = self._action * self.action_scale

        self._state[:2] += self._state[2:4] * self.time_step
        self._state[2:4] += self._action * self.time_step

        change_sign = np.logical_or(self._state[0:2] <= 0, self._state[0:2] >= 10)
        if np.any(change_sign):
            self._state[2:4][change_sign] = - self._state[2:4][change_sign]

        for i in range(self.n_objects):
            obj_idx = 4 * (i + 1)

            if self.random_walk:
                self._state[obj_idx:obj_idx + 2] += self._state[obj_idx + 2:obj_idx + 4] * self.time_step
                self._state[obj_idx:obj_idx + 2] = np.clip(self._state[obj_idx:obj_idx + 2], 2, 10)
                obj_action = np.random.uniform(-1, 1, 2) * 10
                change_sign = np.logical_or(self._state[obj_idx:obj_idx + 2] <= 2,
                                            self._state[obj_idx:obj_idx + 2] >= 10)
                if np.any(change_sign):
                    self._state[obj_idx + 2:obj_idx + 4][change_sign] = - self._state[obj_idx + 2:obj_idx + 4][
                        change_sign]
                self._state[obj_idx + 2:obj_idx + 4] += obj_action * self.time_step
                self._state[obj_idx + 2:obj_idx + 4] = np.clip(self._state[obj_idx + 2:obj_idx + 4], -1, 1)
            else:
                self._state[obj_idx] = self._obj_circle_center[i][0] + self._obj_radius * np.cos(self._time * 2 * np.pi)
                self._state[obj_idx + 1] = self._obj_circle_center[i][1] + self._obj_radius * np.sin(
                    self._time * 2 * np.pi)
                self._state[obj_idx + 2] = -2 * self._obj_radius * np.pi * np.sin(self._time * 2 * np.pi)
                self._state[obj_idx + 3] = 2 * self._obj_radius * np.pi * np.cos(self._time * 2 * np.pi)

        self._time += self.time_step
        reward = - np.linalg.norm(np.array([9., 9.]) - self._state[:2]) / (8 * np.sqrt(2))
        return self._state, reward, False, dict()

    def render(self):
        self._viewer.circle(center=self._state[:2], radius=0.3, color=(0., 0., 255), width=0)

        for i in range(self.n_objects):
            obj_idx = 4 * (i + 1)
            self._viewer.circle(center=self._state[obj_idx:obj_idx + 2], radius=0.3, color=(255., 0., 0), width=0)

        self._viewer.square(center=np.array([9., 9.]), angle=0, edge=0.6, color=(0., 255., 0))

        self._viewer.display(self.time_step * 0.4)

    def _create_sim_state(self):
        return self._state

    def _create_observation(self, state):
        return state