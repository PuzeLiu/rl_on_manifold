import numpy as np
import pinocchio as pino
import pybullet_utils.transformations as transformations
from mushroom_rl.core import MDPInfo
from mushroom_rl.environments.pybullet import PyBulletObservationType
from mushroom_rl.utils.spaces import Box
from atacom.environments.iiwa_air_hockey.env_base import AirHockeyBase
from atacom.environments.iiwa_air_hockey.kinematics import clik, fk


class AirHockeySingle(AirHockeyBase):
    def __init__(self, gamma=0.99, horizon=500, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 env_noise=False, obs_noise=False, obs_delay=False, torque_control=True, step_action_function=None,
                 isolated_joint_7=False):
        self.obs_prev = None

        if isolated_joint_7:
            self.n_ctrl_joints = 6
        else:
            self.n_ctrl_joints = 7

        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep,
                         n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui,
                         env_noise=env_noise, n_agents=1, obs_noise=obs_noise, obs_delay=obs_delay,
                         torque_control=torque_control, step_action_function=step_action_function,
                         isolated_joint_7=isolated_joint_7)

        self._compute_init_state()

        self._client.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-00.0, cameraPitch=-45.0,
                                                cameraTargetPosition=[-0.5, 0., 0.])

        self._change_dynamics()

        self._disable_collision()

        self.reset()

    def _compute_init_state(self):
        q = np.zeros(9)
        des_pos = pino.SE3(np.diag([-1., 1., -1.]), np.array([0.65, 0., self.env_spec['universal_height']]))

        success, self.init_state = clik(self.pino_model, self.pino_data, des_pos, q, self.frame_idx)
        assert success is True

    def _disable_collision(self):
        # disable the collision with left and right rim Because of the improper collision shape
        iiwa_links = ['iiwa_1/link_1', 'iiwa_1/link_2', 'iiwa_1/link_3', 'iiwa_1/link_4', 'iiwa_1/link_5',
                      'iiwa_1/link_6', 'iiwa_1/link_7', 'iiwa_1/link_ee', 'iiwa_1/striker_base',
                      'iiwa_1/striker_joint_link', 'iiwa_1/striker_mallet', 'iiwa_1/striker_mallet_tip']
        table_rims = ['t_down_rim_l', 't_down_rim_r', 't_up_rim_r', 't_up_rim_l',
                      't_left_rim', 't_right_rim', 't_base', 't_up_rim_top', 't_down_rim_top', 't_base']
        for iiwa_l in iiwa_links:
            for table_r in table_rims:
                self.client.setCollisionFilterPair(self._indexer.link_map[iiwa_l][0],
                                                   self._indexer.link_map[table_r][0],
                                                   self._indexer.link_map[iiwa_l][1],
                                                   self._indexer.link_map[table_r][1], 0)

        self.client.setCollisionFilterPair(self._model_map['puck'], self._indexer.link_map['t_down_rim_top'][0],
                                           -1, self._indexer.link_map['t_down_rim_top'][1], 0)
        self.client.setCollisionFilterPair(self._model_map['puck'], self._indexer.link_map['t_up_rim_top'][0],
                                           -1, self._indexer.link_map['t_up_rim_top'][1], 0)

    def _change_dynamics(self):
        for i in range(12):
            self.client.changeDynamics(self._model_map['iiwa_1'], i, linearDamping=0., angularDamping=0.)

    def _modify_mdp_info(self, mdp_info):
        obs_idx = [0, 1, 2, 7, 8, 9, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27]
        obs_low = mdp_info.observation_space.low[obs_idx]
        obs_high = mdp_info.observation_space.high[obs_idx]
        obs_low[0:3] = [-1, -0.5, -np.pi]
        obs_high[0:3] = [1, 0.5, np.pi]
        observation_space = Box(low=obs_low, high=obs_high)

        act_low = mdp_info.action_space.low[:self.n_ctrl_joints]
        act_high = mdp_info.action_space.high[:self.n_ctrl_joints]
        action_space = Box(low=act_low, high=act_high)
        return MDPInfo(observation_space, action_space, mdp_info.gamma, mdp_info.horizon)

    def _create_observation(self, state):
        puck_pose = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_POS)
        puck_pose_2d = self._puck_2d_in_robot_frame(puck_pose, self.agents[0]['frame'], type='pose')

        robot_pos = list()
        robot_vel = list()
        for i in range(6):
            robot_pos.append(self.get_sim_state(state,
                                                self.agents[0]['name'] + "/joint_"+str(i+1),
                                                PyBulletObservationType.JOINT_POS))
            robot_vel.append(self.get_sim_state(state,
                                                self.agents[0]['name'] + "/joint_" + str(i + 1),
                                                PyBulletObservationType.JOINT_VEL))
        if not self.isolated_joint_7:
            robot_pos.append(self.get_sim_state(state,
                                                self.agents[0]['name'] + "/joint_" + str(7),
                                                PyBulletObservationType.JOINT_POS))
            robot_vel.append(self.get_sim_state(state,
                                                self.agents[0]['name'] + "/joint_" + str(7),
                                                PyBulletObservationType.JOINT_VEL))
        robot_pos = np.asarray(robot_pos).flatten()
        robot_vel = np.asarray(robot_vel).flatten()

        if self.obs_noise:
            puck_pose_2d[:2] += np.random.randn(2) * 0.001
            puck_pose_2d[2] += np.random.randn(1) * 0.001

        puck_lin_vel = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_LIN_VEL)
        puck_ang_vel = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_ANG_VEL)
        puck_vel_2d = self._puck_2d_in_robot_frame(np.concatenate([puck_lin_vel, puck_ang_vel]),
                                                   self.agents[0]['frame'], type='vel')

        if self.obs_delay:
            alpha = 0.5
            puck_vel_2d = alpha * puck_vel_2d + (1 - alpha) * self.obs_prev[3:6]
            robot_vel = alpha * robot_vel + (1 - alpha) * self.obs_prev[9:12]

        self.obs_prev = np.concatenate([puck_pose_2d, puck_vel_2d, robot_pos, robot_vel])
        return self.obs_prev

    def _puck_2d_in_robot_frame(self, puck_in, robot_frame, type='pose'):
        if type == 'pose':
            puck_frame = transformations.translation_matrix(puck_in[:3])
            puck_frame = puck_frame @ transformations.quaternion_matrix(puck_in[3:])

            frame_target = transformations.inverse_matrix(robot_frame) @ puck_frame
            puck_translate = transformations.translation_from_matrix(frame_target)
            _, _, puck_euler_yaw = transformations.euler_from_matrix(frame_target)

            return np.concatenate([puck_translate[:2], [puck_euler_yaw]])
        if type == 'vel':
            rot_mat = robot_frame[:3, :3]
            vec_lin = rot_mat.T @ puck_in[:3]
            return np.concatenate([vec_lin[:2], puck_in[5:6]])

    def _compute_joint_7(self, joint_state):
        q_cur = joint_state.copy()
        q_cur_7 = q_cur[6]
        q_cur[6] = 0.

        f_cur = fk(self.pino_model, self.pino_data, q_cur, self.frame_idx)
        z_axis = np.array([0., 0., -1.])

        y_des = np.cross(z_axis, f_cur.rotation[:, 2])
        y_des_norm = np.linalg.norm(y_des)
        if y_des_norm > 1e-2:
            y_des = y_des / y_des_norm
        else:
            y_des = f_cur.rotation[:, 2]

        target = np.arccos(f_cur.rotation[:, 1].dot(y_des))

        axis = np.cross(f_cur.rotation[:, 1], y_des)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-2:
            axis = axis / axis_norm
        else:
            axis = np.array([0., 0., 1.])

        target = target * axis.dot(f_cur.rotation[:, 2])

        if target - q_cur_7 > np.pi / 2:
            target -= np.pi
        elif target - q_cur_7 < -np.pi / 2:
            target += np.pi

        return np.atleast_1d(target)

    def _compute_universal_joint(self, joint_state):
        rot_mat = transformations.quaternion_matrix(
            self.client.getLinkState(*self._indexer.link_map['iiwa_1/link_ee'])[1])

        q1 = np.arccos(rot_mat[:3, 2].dot(np.array((0., 0., -1))))
        q2 = 0

        axis = np.cross(rot_mat[:3, 2], np.array([0., 0., -1.]))
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-2:
            axis = axis / axis_norm
        else:
            axis = np.array([0., 0., 1.])
        q1 = q1 * axis.dot(rot_mat[:3, 1])

        return np.array([q1, q2])
