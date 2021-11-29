import os
import numpy as np
import pinocchio as pino
import matplotlib.pyplot as plt
from atacom.environments.iiwa_air_hockey.env_hitting import AirHockeyHit
from atacom.atacom import AtacomEnvWrapper
from atacom.constraints import ViabilityConstraint, ConstraintsSet


class AirHockeyIiwaAtacom(AtacomEnvWrapper):
    def __init__(self, task='H', gamma=0.99, horizon=120, timestep=1 / 240., n_intermediate_steps=4,
                 debug_gui=False, env_noise=False, obs_noise=False, obs_delay=False, Kc=240., random_init=False,
                 action_penalty=1e-3):
        if task == 'H':
            base_env = AirHockeyHit(gamma=gamma, horizon=horizon, timestep=timestep,
                                    n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                                    env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                                    torque_control='torque', random_init=random_init,
                                    action_penalty=action_penalty, isolated_joint_7=True)
        if task == 'D':
            raise NotImplementedError

        dim_q = base_env.n_ctrl_joints
        ee_pos_f = ViabilityConstraint(dim_q=dim_q, dim_out=1, fun=self.ee_pose_f, J=self.ee_pose_J_f,
                                       b=self.ee_pos_b_f, K=0.1)
        f = ConstraintsSet(dim_q)
        f.add_constraint(ee_pos_f)

        cart_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=5, fun=self.ee_pos_g, J=self.ee_pos_J_g,
                                         b=self.ee_pos_b_g, K=0.5)
        joint_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=dim_q, fun=self.joint_pos_g, J=self.joint_pos_J_g,
                                          b=self.joint_pos_b_g, K=1)
        g = ConstraintsSet(dim_q)
        g.add_constraint(cart_pos_g)
        g.add_constraint(joint_pos_g)

        acc_max = np.ones(base_env.n_ctrl_joints) * 10
        vel_max = base_env.joints.velocity_limits()[:base_env.n_ctrl_joints]
        super().__init__(base_env, dim_q, f=f, g=g, Kc=Kc, vel_max=vel_max, acc_max=acc_max, Kq=4 * acc_max / vel_max,
                         time_step=timestep)

        self.pino_model = self.env.pino_model
        self.pino_data = self.env.pino_data
        self.frame_idx = self.env.frame_idx
        self.frame_idx_4 = 12
        self.frame_idx_7 = 18

        for i in range(self.pino_model.nq):
            self.env.client.changeDynamics(*base_env._indexer.joint_map[self.pino_model.names[i+1]],
                                           maxJointVelocity=self.pino_model.velocityLimit[i] * 1.5)

    def _get_q(self, state):
        return state[-2 * self.env.n_ctrl_joints:-self.env.n_ctrl_joints]

    def _get_dq(self, state):
        return state[-self.env.n_ctrl_joints:]

    def acc_to_ctrl_action(self, ddq):
        ddq = self._get_pino_value(ddq).tolist()
        sim_state = self.env._indexer.create_sim_state()
        q = self.env.joints.positions(sim_state).tolist()
        dq = self.env.joints.velocities(sim_state).tolist()
        return self.env.client.calculateInverseDynamics(2, q, dq, ddq)[:self.env.n_ctrl_joints]

    def _get_pino_value(self, q):
        ret = np.zeros(9)
        ret[:q.shape[0]] = q
        return ret

    def ee_pose_f(self, q):
        q = self._get_pino_value(q)
        pino.framesForwardKinematics(self.pino_model, self.pino_data, q)
        ee_pos_z = self.pino_data.oMf[self.frame_idx].translation[2]
        return np.atleast_1d(ee_pos_z - self.env.env_spec['universal_height'])

    def ee_pose_J_f(self, q):
        q = self._get_pino_value(q)
        pino.framesForwardKinematics(self.pino_model, self.pino_data, q)
        ee_jac = pino.computeFrameJacobian(self.pino_model, self.pino_data, q,
                                           self.frame_idx, pino.LOCAL_WORLD_ALIGNED)[:, :self.env.n_ctrl_joints]
        J_pos = ee_jac[2]
        return np.atleast_2d(J_pos)

    def ee_pos_b_f(self, q, dq):
        q = self._get_pino_value(q)
        dq = self._get_pino_value(dq)
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        acc = pino.getFrameClassicalAcceleration(self.pino_model, self.pino_data, self.frame_idx,
                                                 pino.LOCAL_WORLD_ALIGNED).vector
        b_pos = acc[2]
        return np.atleast_1d(b_pos)

    def ee_pos_g(self, q):
        q = self._get_pino_value(q)
        pino.framesForwardKinematics(self.pino_model, self.pino_data, q)
        ee_pos = self.pino_data.oMf[self.frame_idx].translation[:2]
        ee_pos_world = ee_pos + self.env.agents[0]['frame'][:2, 3]
        g_1 = - ee_pos_world[0] - (self.env.env_spec['table']['length'] / 2 - self.env.env_spec['mallet']['radius'])
        g_2 = - ee_pos_world[1] - (self.env.env_spec['table']['width'] / 2 - self.env.env_spec['mallet']['radius'])
        g_3 = ee_pos_world[1] - (self.env.env_spec['table']['width'] / 2 - self.env.env_spec['mallet']['radius'])

        ee_pos_4 = self.pino_data.oMf[self.frame_idx_4].translation
        ee_pos_7 = self.pino_data.oMf[self.frame_idx_7].translation
        g_4 = -ee_pos_4[2] + 0.36
        g_5 = -ee_pos_7[2] + 0.25
        return np.array([g_1, g_2, g_3, g_4, g_5])

    def ee_pos_J_g(self, q):
        q = self._get_pino_value(q)
        pino.computeJointJacobians(self.pino_model, self.pino_data, q)
        jac_ee = pino.getFrameJacobian(self.pino_model, self.pino_data,
                                        self.frame_idx, pino.LOCAL_WORLD_ALIGNED)[:, :self.env.n_ctrl_joints]
        jac_4 = pino.getFrameJacobian(self.pino_model, self.pino_data,
                                      self.frame_idx_4, pino.LOCAL_WORLD_ALIGNED)[:, :self.env.n_ctrl_joints]
        jac_7 = pino.getFrameJacobian(self.pino_model, self.pino_data,
                                      self.frame_idx_7, pino.LOCAL_WORLD_ALIGNED)[:, :self.env.n_ctrl_joints]
        return np.vstack([-jac_ee[0], -jac_ee[1], jac_ee[1], -jac_4[2], -jac_7[2]])

    def ee_pos_b_g(self, q, dq):
        q = self._get_pino_value(q)
        dq = self._get_pino_value(dq)
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        acc_ee = pino.getFrameClassicalAcceleration(self.pino_model, self.pino_data, self.frame_idx,
                                                 pino.LOCAL_WORLD_ALIGNED).vector
        acc_4 = pino.getFrameClassicalAcceleration(self.pino_model, self.pino_data, self.frame_idx_4,
                                                   pino.LOCAL_WORLD_ALIGNED).vector
        acc_7 = pino.getFrameClassicalAcceleration(self.pino_model, self.pino_data, self.frame_idx_7,
                                                   pino.LOCAL_WORLD_ALIGNED).vector

        return np.array([-acc_ee[0], -acc_ee[1], acc_ee[1], -acc_4[2], -acc_7[2]])

    def joint_pos_g(self, q):
        return np.array(q ** 2 - self.pino_model.upperPositionLimit[:self.env.n_ctrl_joints] ** 2)

    def joint_pos_J_g(self, q):
        return 2 * np.diag(q)

    def joint_pos_b_g(self, q, dq):
        return 2 * dq ** 2

    def plot_constraints(self, dataset, save_dir="", suffix="", state_norm_processor=None):
        state_list = list()
        i = 0

        if suffix != '':
            suffix = suffix + "_"

        for data in dataset:
            state = data[0]
            if state_norm_processor is not None:
                state[state_norm_processor._obs_mask] = (state * state_norm_processor._obs_delta + \
                                                        state_norm_processor._obs_mean)[state_norm_processor._obs_mask]
            state_list.append(state)
            if data[-1]:
                i += 1
                state_hist = np.array(state_list)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                ee_pos_list = list()
                for state_i in state_hist:
                    q = np.zeros(9)
                    q[:6] = state_i[6:12]
                    pino.framesForwardKinematics(self.pino_model, self.pino_data, q)
                    ee_pos_list.append(self.pino_data.oMf[-1].translation[:2] + self.env.agents[0]['frame'][:2, 3])

                ee_pos_list = np.array(ee_pos_list)
                fig1, axes1 = plt.subplots(1, figsize=(10, 10))
                axes1.plot(ee_pos_list[:, 0], ee_pos_list[:, 1], label='position')
                axes1.plot([0.0, -0.91, -0.91, 0.0], [-0.45, -0.45, 0.45, 0.45], label='boundary', c='k', lw='5')
                axes1.set_aspect(1.0)
                axes1.set_xlim(-1, 0)
                axes1.set_ylim(-0.5, 0.5)
                axes1.legend(loc='upper right')
                axes1.set_title('EndEffector')
                axes1.legend(loc='center right')
                file1 = "EndEffector_" + suffix + str(i) + ".pdf"
                plt.savefig(os.path.join(save_dir, file1))
                plt.close(fig1)

                fig2, axes2 = plt.subplots(2, 3, figsize=(21, 8), sharex=True, sharey=True)
                for j in range(6):
                    axes2[j // 3, j % 3].plot(state_hist[:, 6 + j], lw=3, color='tab:blue')
                    axes2[j // 3, j % 3].plot(state_hist[:, 12 + j], lw=3, color='tab:orange')
                    axes2[j // 3, j % 3].plot([0, state_hist.shape[0]], [self.pino_model.lowerPositionLimit[j]] * 2,
                                              lw=3, c='tab:red', ls='--')
                    axes2[j // 3, j % 3].plot([0, state_hist.shape[0]], [self.pino_model.upperPositionLimit[j]] * 2,
                                              lw=3, c='tab:red', ls='--')
                    axes2[j // 3, j % 3].plot([0, state_hist.shape[0]], [-self.pino_model.velocityLimit[j]] * 2,
                                              lw=3, c='tab:pink', ls=':')
                    axes2[j // 3, j % 3].plot([0, state_hist.shape[0]], [self.pino_model.velocityLimit[j]] * 2,
                                              lw=3, c='tab:pink', ls=':')

                    axes2[j // 3, j % 3].set_title('Joint ' + str(j + 1))

                axes2[0, 0].plot([], lw=3, color='tab:blue', label='position')
                axes2[0, 0].plot([], lw=3, color='tab:red', ls='--', label='position limit')
                axes2[0, 0].plot([], lw=3, color='tab:orange', label='velocity')
                axes2[0, 0].plot([], lw=3, color='tab:pink', ls=':', label='velocity limit')
                fig2.legend(ncol=4, loc='lower center')

                file2 = "JointProfile_" + suffix + str(i) + ".pdf"
                plt.savefig(os.path.join(save_dir, file2))
                plt.close(fig2)

                state_list = list()




