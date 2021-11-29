import time
import pinocchio as pino
from mushroom_rl.utils.spaces import *
from scipy.linalg import solve
from atacom.environments.iiwa_air_hockey.env_hitting import AirHockeyHit


class AirHockeyIiwaRmp:
    def __init__(self, task='H', gamma=0.99, horizon=120, timestep=1 / 240., n_intermediate_steps=4,
                 acc_max=10, debug_gui=False, env_noise=False, obs_noise=False, obs_delay=False, random_init=False,
                 action_penalty=1e-3, Kq=10.):
        if task == 'H':
            base_env = AirHockeyHit(gamma=gamma, horizon=horizon, timestep=timestep,
                                    n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                                    env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                                    torque_control='torque', random_init=random_init,
                                    step_action_function=self.step_action_function,
                                    action_penalty=action_penalty, isolated_joint_7=True)
        if task == 'D':
            raise NotImplementedError

        self.observation = None

        self.env = base_env
        self.env.agents[0]['constr_a'] = 1.0
        self.env.agents[0]['constr_b'] = 1.0
        self.env.agents[0]['constr_bound_x_l'] = 0.58
        self.env.agents[0]['constr_bound_y_l'] = -0.46
        self.env.agents[0]['constr_bound_y_u'] = 0.46

        self.dim_q = 6
        self._mdp_info = self.env.info.copy()
        self._mdp_info.action_space = Box(low=-np.ones(self.dim_q), high=np.ones(self.dim_q))

        self.constr_logs = list()

        if np.isscalar(acc_max):
            self.acc_max = np.ones(self.dim_q) * acc_max
        else:
            self.acc_max = acc_max
            assert np.shape(self.acc_max)[0] == self.dim_q

        if np.isscalar(Kq):
            self.K_q = np.ones(self.dim_q) * Kq
        else:
            self.K_q = Kq
            assert np.shape(self.K_q)[0] == self.dim_q

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.
        """
        return self._mdp_info

    def reset(self, state=None):
        return self.env.reset(state)

    def step(self, action):
        action = np.clip(action, self.info.action_space.low, self.info.action_space.high)
        self.observation, reward, absorbing, _ = self.env.step(action)
        q = self.env.joints.positions(self.env._state)
        dq = self.env.joints.positions(self.env._state)
        self._update_constraint_stats(q, dq)
        return self.observation.copy(), reward, absorbing, _

    def stop(self):
        self.env.stop()

    def step_action_function(self, state, action):
        q = self.env.joints.positions(state)
        dq = self.env.joints.velocities(state)
        pino.forwardKinematics(self.env.pino_model, self.env.pino_data, q, dq)
        pino.computeJointJacobians(self.env.pino_model, self.env.pino_data, q)
        pino.updateFramePlacements(self.env.pino_model, self.env.pino_data)

        ddq_ee = self.rmp_ddq_ee()
        ddq_elbow = self.rmp_ddq_elbow()
        ddq_wrist = self.rmp_ddq_wrist()

        ddq_joints = self.rmp_joint_limit(q, dq)

        ddq_total = np.zeros(self.env.pino_model.nq)
        ddq_total[:self.dim_q] = ddq_ee + ddq_elbow + ddq_wrist + ddq_joints + action * self.acc_max
        ddq = self.acc_truncation(dq, ddq_total)
        tau = pino.rnea(self.env.pino_model, self.env.pino_data, q, dq, ddq)
        return tau[:6]

    def rmp_ddq_ee(self):
        frame_id = self.env.frame_idx
        J_frame = pino.getFrameJacobian(self.env.pino_model, self.env.pino_data, frame_id,
                                        pino.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_frame[:3, :6]
        link_pos = self.env.pino_data.oMf[frame_id].translation
        link_vel = pino.getFrameVelocity(self.env.pino_model, self.env.pino_data, frame_id,
                                         pino.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        cart_acc = self.f_bound(link_pos, link_vel, self.env.agents[0]['constr_bound_x_l'], idx=0, b_type='l',
                                eta_rep=0.1, v_rep=10) + \
                   self.f_bound(link_pos, link_vel, self.env.agents[0]['constr_bound_y_l'], idx=1, b_type='l',
                                eta_rep=0.1, v_rep=10) + \
                   self.f_bound(link_pos, link_vel, self.env.agents[0]['constr_bound_y_u'], idx=1, b_type='u',
                                eta_rep=0.1, v_rep=10) + \
                   self.f_plane(link_pos, link_vel, self.env.env_spec['universal_height'])

        return J.T @ solve(J @ J.T + 1e-6 * np.eye(3), cart_acc)

    def rmp_ddq_elbow(self):
        frame_id = self.env.pino_model.getFrameId("iiwa_1/link_4")
        J_frame = pino.getFrameJacobian(self.env.pino_model, self.env.pino_data, frame_id,
                                        pino.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_frame[:3, :6]
        link_pos = self.env.pino_data.oMf[frame_id].translation
        link_vel = pino.getFrameVelocity(self.env.pino_model, self.env.pino_data, frame_id,
                                         pino.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        cart_acc = self.f_bound(link_pos, link_vel, 0.36, idx=2, b_type='l',
                                eta_rep=0.5, v_rep=10)
        return J.T @ solve(J @ J.T + 1e-6 * np.eye(3), cart_acc)

    def rmp_ddq_wrist(self):
        frame_id = self.env.pino_model.getFrameId("iiwa_1/link_6")
        J_frame = pino.getFrameJacobian(self.env.pino_model, self.env.pino_data, frame_id,
                                        pino.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_frame[:3, :6]
        link_pos = self.env.pino_data.oMf[frame_id].translation
        link_vel = pino.getFrameVelocity(self.env.pino_model, self.env.pino_data, frame_id,
                                         pino.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        cart_acc = self.f_bound(link_pos, link_vel, 0.25, idx=2, b_type='l',
                                eta_rep=0.01, v_rep=10)
        return J.T @ solve(J @ J.T + 1e-6 * np.eye(3), cart_acc)

    def rmp_joint_limit(self, q, dq):
        sigma = 10

        s = (q - self.env.pino_model.lowerPositionLimit) / \
            (self.env.pino_model.upperPositionLimit - self.env.pino_model.lowerPositionLimit)

        d = 4 * s * (1 - s)

        alpha_u = 1 - np.exp(- (np.maximum(dq, 0.) / sigma) ** 2 / 2)
        alpha_l = 1 - np.exp(- (np.minimum(dq, 0.) / sigma) ** 2 / 2)

        b = s * (alpha_u * d + (1 - alpha_u)) + (1 - s) * (alpha_l * d + (1 - alpha_l))

        a = b ** (-2)
        return a[:6]

    def f_bound(self, x, dx, bound, idx, b_type='u', eta_rep=5., v_rep=1., eta_damp=None):
        ddx = np.zeros_like(x)
        if eta_damp is None:
            eta_damp = np.sqrt(eta_rep)
        if b_type == 'u':
            d = np.maximum(bound - x[idx], 0.) ** 2
            ddx[idx] = -eta_rep * np.exp(-d / v_rep) - eta_damp / (d + 1e-6) * np.maximum(dx[idx], 0)
        elif b_type == 'l':
            d = np.maximum(x[idx] - bound, 0.) ** 2
            ddx[idx] = eta_rep * np.exp(-d / v_rep) - eta_damp / (d + 1e-6) * np.minimum(dx[idx], 0)
        return ddx

    def f_plane(self, x, dx, height):
        ddx = np.zeros_like(x)
        k = 1000
        d = np.sqrt(k)
        ddx[2] = k * (height - x[2]) - d * dx[2]
        return ddx

    def acc_truncation(self, dq, ddq):
        acc_u = np.maximum(np.minimum(self.acc_max,
                                      -self.K_q * (dq[:self.dim_q] - self.env.pino_model.velocityLimit[:self.dim_q])),
                           -self.acc_max)
        acc_l = np.minimum(np.maximum(-self.acc_max,
                                      -self.K_q * (dq[:self.dim_q] + self.env.pino_model.velocityLimit[:self.dim_q])),
                           self.acc_max)
        ddq[:self.dim_q] = np.clip(ddq[:self.dim_q], acc_l, acc_u)
        return ddq

    def _update_constraint_stats(self, q, dq):
        c_i = self._compute_c(q, dq)
        c_dq_i = (np.abs(dq) - self.env.pino_model.velocityLimit)[:self.dim_q]
        self.constr_logs.append([np.max(c_i), np.max(c_dq_i)])

    def get_constraints_logs(self):
        constr_logs = np.array(self.constr_logs)
        c_avg = np.mean(constr_logs[:, 0])
        c_max = np.max(constr_logs[:, 0])
        c_dq_max = np.max(constr_logs[:, 1])
        self.constr_logs.clear()
        return c_avg, c_max, c_dq_max

    def _compute_c(self, q, dq):
        pino.forwardKinematics(self.env.pino_model, self.env.pino_data, q, dq)
        pino.computeJointJacobians(self.env.pino_model, self.env.pino_data, q)
        pino.updateFramePlacements(self.env.pino_model, self.env.pino_data)

        ee_pos = self.env.pino_data.oMf[self.env.frame_idx].translation
        elbow_pos = self.env.pino_data.oMf[self.env.pino_model.getFrameId("iiwa_1/link_4")].translation
        wrist_pos = self.env.pino_data.oMf[self.env.pino_model.getFrameId("iiwa_1/link_6")].translation

        c = []
        c.append(np.abs(ee_pos[2] - self.env.env_spec['universal_height']))
        c.append(-ee_pos[0] + self.env.agents[0]['constr_bound_x_l'])
        c.append(-ee_pos[1] + self.env.agents[0]['constr_bound_y_l'])
        c.append(ee_pos[1] - self.env.agents[0]['constr_bound_y_u'])
        c.append(- elbow_pos[2] + 0.36)
        c.append(- wrist_pos[2] + 0.25)
        c.extend(- q[:self.env.n_ctrl_joints] + self.env.pino_model.lowerPositionLimit[:self.env.n_ctrl_joints])
        c.extend(q[:self.env.n_ctrl_joints] - self.env.pino_model.upperPositionLimit[:self.env.n_ctrl_joints])
        return np.array(c)


if __name__ == '__main__':
    env = AirHockeyIiwaRmp(debug_gui=True)

    env.reset()
    for i in range(10000):
        action = np.random.randn(env.dim_q)
        _, _, absorb, _ = env.step(action)
        if absorb:
            env.reset()
        time.sleep(1 / 240.)
