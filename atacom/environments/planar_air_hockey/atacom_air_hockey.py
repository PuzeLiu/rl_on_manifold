import os
import numpy as np
import pinocchio as pino
import matplotlib.pyplot as plt

from atacom.atacom import AtacomEnvWrapper
from atacom.constraints import ViabilityConstraint, ConstraintsSet
from mushroom_rl.environments.pybullet_envs.air_hockey import AirHockeyHit, AirHockeyDefend


class AirHockeyPlanarAtacom(AtacomEnvWrapper):
    def __init__(self, task='H', gamma=0.99, horizon=120, timestep=1 / 240., n_intermediate_steps=4,
                 debug_gui=False, env_noise=False, obs_noise=False, obs_delay=False, Kc=240., random_init=False,
                 action_penalty=1e-3):
        if task == 'H':
            base_env = AirHockeyHit(gamma=gamma, horizon=horizon, timestep=timestep,
                                    n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                                    env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                                    torque_control=True, random_init=random_init,
                                    action_penalty=action_penalty)
        if task == 'D':
            base_env = AirHockeyDefend(gamma=gamma, horizon=horizon, timestep=timestep,
                                       n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                                       env_noise=env_noise, obs_noise=obs_noise, obs_delay=obs_delay,
                                       torque_control=True, random_init=random_init,
                                       action_penalty=action_penalty)

        dim_q = 3
        cart_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=3, fun=self.cart_pos_g, J=self.cart_pos_J_g,
                                         b=self.cart_pos_b_g, K=0.5)
        joint_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=3, fun=self.joint_pos_g, J=self.joint_pos_J_g,
                                          b=self.joint_pos_b_g, K=1.0)
        # joint_vel_g = StateVelocityConstraint(dim_q=dim_q, dim_out=3, fun=self.joint_vel_g, A=self.joint_vel_A_g,
        #                                       b=self.joint_vel_b_g, margin=0.0)
        g = ConstraintsSet(dim_q)
        g.add_constraint(cart_pos_g)
        g.add_constraint(joint_pos_g)
        # g.add_constraint(joint_vel_g)

        acc_max = np.ones(3) * 10
        vel_max = base_env.joints.velocity_limits()
        super().__init__(base_env, 3, f=None, g=g, Kc=Kc, vel_max=vel_max, acc_max=acc_max, Kq=2 * acc_max / vel_max,
                         time_step=timestep)

        self.pino_model = pino.buildModelFromUrdf(self.env.agents[0]['urdf'])
        self.pino_data = self.pino_model.createData()
        self.frame_idx = self.pino_model.nframes - 1

        self.env.client.changeDynamics(*base_env._indexer.joint_map[self.pino_model.names[1]],
                                       maxJointVelocity=self.pino_model.velocityLimit[0] * 1.5)
        self.env.client.changeDynamics(*base_env._indexer.joint_map[self.pino_model.names[2]],
                                       maxJointVelocity=self.pino_model.velocityLimit[1] * 1.5)
        self.env.client.changeDynamics(*base_env._indexer.joint_map[self.pino_model.names[3]],
                                       maxJointVelocity=self.pino_model.velocityLimit[2] * 1.5)

        robot_links = ['planar_robot_1/link_striker_hand', 'planar_robot_1/link_striker_ee']
        table_rims = ['t_down_rim_l', 't_down_rim_r', 't_up_rim_r', 't_up_rim_l',
                      't_left_rim', 't_right_rim', 't_base', 't_up_rim_top', 't_down_rim_top', 't_base']
        for iiwa_l in robot_links:
            for table_r in table_rims:
                self.env.client.setCollisionFilterPair(self.env._indexer.link_map[iiwa_l][0],
                                                       self.env._indexer.link_map[table_r][0],
                                                       self.env._indexer.link_map[iiwa_l][1],
                                                       self.env._indexer.link_map[table_r][1], 0)

    def _get_q(self, state):
        return state[6:9]

    def _get_dq(self, state):
        return state[9:12]

    def acc_to_ctrl_action(self, ddq):
        q = self.q.tolist()
        dq = self.dq.tolist()
        ddq = ddq.tolist()
        return self.env.client.calculateInverseDynamics(self.env._model_map['planar_robot_1'], q, dq, ddq)

    def cart_pos_g(self, q):
        pino.framesForwardKinematics(self.pino_model, self.pino_data, q)
        ee_pos = self.pino_data.oMf[-1].translation[:2]
        ee_pos_world = ee_pos + self.env.agents[0]['frame'][:2, 3]
        g_1 = - ee_pos_world[0] - (self.env.env_spec['table']['length'] / 2 - self.env.env_spec['mallet']['radius'])
        g_2 = - ee_pos_world[1] - (self.env.env_spec['table']['width'] / 2 - self.env.env_spec['mallet']['radius'])
        g_3 = ee_pos_world[1] - (self.env.env_spec['table']['width'] / 2 - self.env.env_spec['mallet']['radius'])
        return np.array([g_1, g_2, g_3])

    def cart_pos_J_g(self, q):
        ee_jac = pino.computeFrameJacobian(self.pino_model, self.pino_data, q,
                                           self.frame_idx, pino.LOCAL_WORLD_ALIGNED)[:2]
        J_c = np.array([[-1., 0.], [0., -1.], [0., 1.]])
        return J_c @ ee_jac

    def cart_pos_b_g(self, q, dq):
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        acc = pino.getFrameClassicalAcceleration(self.pino_model, self.pino_data, self.pino_model.nframes - 1,
                                                 pino.LOCAL_WORLD_ALIGNED).vector
        J_c = np.array([[-1., 0.], [0., -1.], [0., 1.]])
        return J_c @ acc[:2]

    def joint_pos_g(self, q):
        return np.array(q ** 2 - self.pino_model.upperPositionLimit ** 2)

    def joint_pos_J_g(self, q):
        return 2 * np.diag(q)

    def joint_pos_b_g(self, q, dq):
        return 2 * dq ** 2

    def joint_vel_g(self, q, dq):
        return np.array([dq ** 2 - self.pino_model.velocityLimit ** 2])

    def joint_vel_A_g(self, q, dq):
        return 2 * np.diag(dq)

    def joint_vel_b_g(self, q, dq):
        return np.zeros(3)

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
                    pino.framesForwardKinematics(self.pino_model, self.pino_data, state_i[6:9])
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

                fig2, axes2 = plt.subplots(1, 3, sharey=True, figsize=(21, 8))
                axes2[0].plot(state_hist[:, 6], label='position', c='tab:blue')
                axes2[1].plot(state_hist[:, 7], c='tab:blue')
                axes2[2].plot(state_hist[:, 8], c='tab:blue')
                axes2[0].plot([0, state_hist.shape[0]], [self.pino_model.lowerPositionLimit[0]] * 2,
                              label='position limit', c='tab:red', ls='--')
                axes2[1].plot([0, state_hist.shape[0]], [self.pino_model.lowerPositionLimit[1]] * 2, c='tab:red',
                              ls='--')
                axes2[2].plot([0, state_hist.shape[0]], [self.pino_model.lowerPositionLimit[2]] * 2, c='tab:red',
                              ls='--')
                axes2[0].plot([0, state_hist.shape[0]], [self.pino_model.upperPositionLimit[0]] * 2, c='tab:red',
                              ls='--')
                axes2[1].plot([0, state_hist.shape[0]], [self.pino_model.upperPositionLimit[1]] * 2, c='tab:red',
                              ls='--')
                axes2[2].plot([0, state_hist.shape[0]], [self.pino_model.upperPositionLimit[2]] * 2, c='tab:red',
                              ls='--')

                axes2[0].plot(state_hist[:, 9], label='velocity', c='tab:orange')
                axes2[1].plot(state_hist[:, 10], c='tab:orange')
                axes2[2].plot(state_hist[:, 11], c='tab:orange')
                axes2[0].plot([0, state_hist.shape[0]], [-self.pino_model.velocityLimit[0]] * 2,
                              label='velocity limit', c='tab:pink', ls=':')
                axes2[1].plot([0, state_hist.shape[0]], [-self.pino_model.velocityLimit[1]] * 2, c='tab:pink', ls=':')
                axes2[2].plot([0, state_hist.shape[0]], [-self.pino_model.velocityLimit[2]] * 2, c='tab:pink', ls=':')
                axes2[0].plot([0, state_hist.shape[0]], [self.pino_model.velocityLimit[0]] * 2, c='tab:pink', ls=':')
                axes2[1].plot([0, state_hist.shape[0]], [self.pino_model.velocityLimit[1]] * 2, c='tab:pink', ls=':')
                axes2[2].plot([0, state_hist.shape[0]], [self.pino_model.velocityLimit[2]] * 2, c='tab:pink', ls=':')

                axes2[0].set_title('Joint 1')
                axes2[1].set_title('Joint 2')
                axes2[2].set_title('Joint 3')
                fig2.legend(ncol=4, loc='lower center')

                file2 = "JointProfile_" + suffix + str(i) + ".pdf"
                plt.savefig(os.path.join(save_dir, file2))
                plt.close(fig2)

                state_list = list()