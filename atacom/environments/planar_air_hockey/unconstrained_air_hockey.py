import numpy as np
from mushroom_rl.environments.pybullet import PyBulletObservationType
from mushroom_rl.environments.pybullet_envs.air_hockey import AirHockeyHit, AirHockeyDefend


class AirHockeyHitUnconstrained(AirHockeyHit):
    def __init__(self, gamma=0.99, horizon=120, timestep=1 / 240., n_intermediate_steps=1,
                 debug_gui=False, env_noise=False, obs_noise=False, obs_delay=False, torque_control="torque",
                 random_init=False, action_penalty=1e-3):
        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep,
                         n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                         env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, random_init=random_init,
                         step_action_function=self._step_action_function,
                         action_penalty=action_penalty)
        self.constr_logs = list()

        self.client.changeDynamics(*self._indexer.link_map["planar_robot_1/link_1"],
                                   maxJointVelocity=self.joints.velocity_limits()[0] * 1.5)
        self.client.changeDynamics(*self._indexer.link_map["planar_robot_1/link_2"],
                                   maxJointVelocity=self.joints.velocity_limits()[1] * 1.5)
        self.client.changeDynamics(*self._indexer.link_map["planar_robot_1/link_3"],
                                   maxJointVelocity=self.joints.velocity_limits()[2] * 1.5)
        self.acc_max = np.ones(3) * 10

    def setup(self, state):
        super().setup(state)
        self.constr_logs.clear()

    def _step_action_function(self, state, action):
        action = np.clip(action, self.info.action_space.low, self.info.action_space.high)
        self._update_constraint_stats(state)
        return action

    def _update_constraint_stats(self, state):
        q = self.joints.positions(state)
        dq = self.joints.velocities(state)
        mallet_pose = self.get_sim_state(state, "planar_robot_1/link_striker_ee", PyBulletObservationType.LINK_POS)
        c_ee_i = np.array([-mallet_pose[0] - self.env_spec['table']['length'] / 2,
                           -mallet_pose[1] - self.env_spec['table']['width'] / 2,
                            mallet_pose[1] - self.env_spec['table']['width'] / 2])
        c_q_i = q ** 2 - self.joints.limits()[1] ** 2
        c_dq_i = dq ** 2 - self.joints.velocity_limits() ** 2
        c_i = np.concatenate([c_ee_i, c_q_i])
        self.constr_logs.append([np.max(c_i), np.max(c_dq_i)])

    def get_constraints_logs(self):
        constr_logs = np.array(self.constr_logs)
        c_avg = np.mean(constr_logs[:, 0])
        c_max = np.max(constr_logs[:, 0])
        c_dq_max = np.max(constr_logs[:, 1])
        self.constr_logs.clear()
        return c_avg, c_max, c_dq_max


class AirHockeyDefendUnconstrained(AirHockeyDefend):
    def __init__(self, gamma=0.99, horizon=120, timestep=1 / 240., n_intermediate_steps=1,
                 debug_gui=False, env_noise=False, obs_noise=False, obs_delay=False, torque_control="torque",
                 random_init=False, action_penalty=1e-3):
        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep,
                         n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                         env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, random_init=random_init,
                         step_action_function=self._step_action_function,
                         action_penalty=action_penalty)
        self.constr_logs = list()

        self.client.changeDynamics(*self._indexer.link_map["planar_robot_1/link_1"],
                                   maxJointVelocity=self.joints.velocity_limits()[0] * 1.5)
        self.client.changeDynamics(*self._indexer.link_map["planar_robot_1/link_2"],
                                   maxJointVelocity=self.joints.velocity_limits()[1] * 1.5)
        self.client.changeDynamics(*self._indexer.link_map["planar_robot_1/link_3"],
                                   maxJointVelocity=self.joints.velocity_limits()[2] * 1.5)
        self.acc_max = np.ones(3) * 10

    def setup(self, state):
        super().setup(state)
        self.constr_logs.clear()

    def _step_action_function(self, state, action):
        action = np.clip(action, self.info.action_space.low, self.info.action_space.high)
        self._update_constraint_stats(state)
        return action

    def _update_constraint_stats(self, state):
        q = self.joints.positions(state)
        dq = self.joints.velocities(state)
        mallet_pose = self.get_sim_state(state, "planar_robot_1/link_striker_ee", PyBulletObservationType.LINK_POS)
        c_ee_i = np.array([-mallet_pose[0] - self.env_spec['table']['length'] / 2,
                           -mallet_pose[1] - self.env_spec['table']['width'] / 2,
                            mallet_pose[1] - self.env_spec['table']['width'] / 2])
        c_q_i = q ** 2 - self.joints.limits()[1] ** 2
        c_dq_i = dq ** 2 - self.joints.velocity_limits() ** 2
        c_i = np.concatenate([c_ee_i, c_q_i])
        self.constr_logs.append([np.max(c_i), np.max(c_dq_i)])

    def get_constraints_logs(self):
        constr_logs = np.array(self.constr_logs)
        c_avg = np.mean(constr_logs[:, 0])
        c_max = np.max(constr_logs[:, 0])
        c_dq_max = np.max(constr_logs[:, 1])
        self.constr_logs.clear()
        return c_avg, c_max, c_dq_max