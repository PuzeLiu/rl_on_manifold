import os
import numpy as np
import pinocchio as pino
import pybullet
import pybullet_utils.transformations as transformations
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType
from mushroom_rl.environments.pybullet_envs import __file__ as env_path


class AirHockeyBase(PyBullet):
    def __init__(self, gamma=0.99, horizon=500, timestep=1 / 240., n_intermediate_steps=1, debug_gui=False,
                 n_agents=1, env_noise=False, obs_noise=False, obs_delay=False, torque_control=True,
                 step_action_function=None, isolated_joint_7=False):
        self.n_agents = n_agents
        self.env_noise = env_noise
        self.obs_noise = obs_noise
        self.obs_delay = obs_delay
        self.step_action_function = step_action_function
        self.isolated_joint_7 = isolated_joint_7

        puck_file = os.path.join(os.path.dirname(os.path.abspath(env_path)),
                                 "data", "air_hockey", "puck.urdf")
        table_file = os.path.join(os.path.dirname(os.path.abspath(env_path)),
                                  "data", "air_hockey", "air_hockey_table.urdf")
        robot_file_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "iiwa_1.urdf")
        robot_file_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "iiwa_2.urdf")

        model_files = dict()
        model_files[puck_file] = dict(flags=pybullet.URDF_USE_IMPLICIT_CYLINDER,
                                      basePosition=[0.0, 0, 0], baseOrientation=[0, 0, 0.0, 1.0])
        model_files[table_file] = dict(useFixedBase=True, basePosition=[0.0, 0, 0],
                                       baseOrientation=[0, 0, 0.0, 1.0])

        actuation_spec = list()
        observation_spec = [("puck", PyBulletObservationType.BODY_POS),
                            ("puck", PyBulletObservationType.BODY_LIN_VEL),
                            ("puck", PyBulletObservationType.BODY_ANG_VEL)]
        self.agents = []

        if torque_control:
            control = pybullet.TORQUE_CONTROL
        else:
            control = pybullet.POSITION_CONTROL

        if 1 <= self.n_agents <= 2:
            agent_spec = dict()
            agent_spec['name'] = "iiwa_1"
            agent_spec.update({"urdf": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "urdf", "iiwa_1.urdf")})
            translate = [-1.51, 0, -0.1]
            quaternion = [0.0, 0.0, 0.0, 1.0]
            agent_spec['frame'] = transformations.translation_matrix(translate)
            agent_spec['frame'] = agent_spec['frame'] @ transformations.quaternion_matrix(quaternion)
            model_files[robot_file_1] = dict(
                flags=pybullet.URDF_USE_IMPLICIT_CYLINDER | pybullet.URDF_USE_INERTIA_FROM_FILE,
                basePosition=translate, baseOrientation=quaternion)

            self.agents.append(agent_spec)
            actuation_spec += [("iiwa_1/joint_1", control),
                               ("iiwa_1/joint_2", control),
                               ("iiwa_1/joint_3", control),
                               ("iiwa_1/joint_4", control),
                               ("iiwa_1/joint_5", control),
                               ("iiwa_1/joint_6", control)]
            if self.isolated_joint_7:
                actuation_spec += [("iiwa_1/joint_7", pybullet.POSITION_CONTROL)]
            else:
                actuation_spec += [("iiwa_1/joint_7", control)]
            actuation_spec += [("iiwa_1/striker_joint_1", pybullet.POSITION_CONTROL),
                               ("iiwa_1/striker_joint_2", pybullet.POSITION_CONTROL)]

            observation_spec += [("iiwa_1/joint_1", PyBulletObservationType.JOINT_POS),
                                 ("iiwa_1/joint_2", PyBulletObservationType.JOINT_POS),
                                 ("iiwa_1/joint_3", PyBulletObservationType.JOINT_POS),
                                 ("iiwa_1/joint_4", PyBulletObservationType.JOINT_POS),
                                 ("iiwa_1/joint_5", PyBulletObservationType.JOINT_POS),
                                 ("iiwa_1/joint_6", PyBulletObservationType.JOINT_POS),
                                 ("iiwa_1/joint_7", PyBulletObservationType.JOINT_POS),
                                 ("iiwa_1/striker_joint_1", PyBulletObservationType.JOINT_POS),
                                 ("iiwa_1/striker_joint_2", PyBulletObservationType.JOINT_POS),
                                 ("iiwa_1/joint_1", PyBulletObservationType.JOINT_VEL),
                                 ("iiwa_1/joint_2", PyBulletObservationType.JOINT_VEL),
                                 ("iiwa_1/joint_3", PyBulletObservationType.JOINT_VEL),
                                 ("iiwa_1/joint_4", PyBulletObservationType.JOINT_VEL),
                                 ("iiwa_1/joint_5", PyBulletObservationType.JOINT_VEL),
                                 ("iiwa_1/joint_6", PyBulletObservationType.JOINT_VEL),
                                 ("iiwa_1/joint_7", PyBulletObservationType.JOINT_VEL),
                                 ("iiwa_1/striker_joint_1", PyBulletObservationType.JOINT_VEL),
                                 ("iiwa_1/striker_joint_2", PyBulletObservationType.JOINT_VEL),
                                 ("iiwa_1/striker_mallet_tip", PyBulletObservationType.LINK_POS),
                                 ("iiwa_1/striker_mallet_tip", PyBulletObservationType.LINK_LIN_VEL)]

            if self.n_agents == 2:
                agent_spec = dict()
                agent_spec['name'] = "iiwa_2"
                agent_spec.update({"urdf": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "urdf", "iiwa_pino.urdf")})
                translate = [1.51, 0, -0.1]
                quaternion = [0.0, 0.0, 1.0, 0.0]
                agent_spec['frame'] = transformations.translation_matrix(translate)
                agent_spec['frame'] = agent_spec['frame'] @ transformations.quaternion_matrix(quaternion)
                model_files[robot_file_2] = dict(
                    flags=pybullet.URDF_USE_IMPLICIT_CYLINDER | pybullet.URDF_USE_INERTIA_FROM_FILE,
                    basePosition=translate, baseOrientation=quaternion)
                self.agents.append(agent_spec)

                actuation_spec += [("iiwa_2/joint_1", control),
                                   ("iiwa_2/joint_2", control),
                                   ("iiwa_2/joint_3", control),
                                   ("iiwa_2/joint_4", control),
                                   ("iiwa_2/joint_5", control),
                                   ("iiwa_2/joint_6", control)]
                if self.isolated_joint_7:
                    actuation_spec += [("iiwa_2/joint_7", pybullet.POSITION_CONTROL)]
                else:
                    actuation_spec += [("iiwa_2/joint_7", control)]
                actuation_spec += [("iiwa_2/striker_joint_1", pybullet.POSITION_CONTROL),
                                   ("iiwa_2/striker_joint_2", pybullet.POSITION_CONTROL)]

                observation_spec += [("iiwa_2/joint_1", PyBulletObservationType.JOINT_POS),
                                     ("iiwa_2/joint_2", PyBulletObservationType.JOINT_POS),
                                     ("iiwa_2/joint_3", PyBulletObservationType.JOINT_POS),
                                     ("iiwa_2/joint_4", PyBulletObservationType.JOINT_POS),
                                     ("iiwa_2/joint_5", PyBulletObservationType.JOINT_POS),
                                     ("iiwa_2/joint_6", PyBulletObservationType.JOINT_POS),
                                     ("iiwa_2/joint_7", PyBulletObservationType.JOINT_POS),
                                     ("iiwa_2/striker_joint_1", PyBulletObservationType.JOINT_POS),
                                     ("iiwa_2/striker_joint_2", PyBulletObservationType.JOINT_POS),
                                     ("iiwa_2/joint_1", PyBulletObservationType.JOINT_VEL),
                                     ("iiwa_2/joint_2", PyBulletObservationType.JOINT_VEL),
                                     ("iiwa_2/joint_3", PyBulletObservationType.JOINT_VEL),
                                     ("iiwa_2/joint_4", PyBulletObservationType.JOINT_VEL),
                                     ("iiwa_2/joint_5", PyBulletObservationType.JOINT_VEL),
                                     ("iiwa_2/joint_6", PyBulletObservationType.JOINT_VEL),
                                     ("iiwa_2/joint_7", PyBulletObservationType.JOINT_VEL),
                                     ("iiwa_2/striker_joint_1", PyBulletObservationType.JOINT_VEL),
                                     ("iiwa_2/striker_joint_2", PyBulletObservationType.JOINT_VEL),
                                     ("iiwa_2/striker_mallet_tip", PyBulletObservationType.LINK_POS),
                                     ("iiwa_2/striker_mallet_tip", PyBulletObservationType.LINK_LIN_VEL)]
        else:
            raise ValueError('n_agents should be 1 or 2')

        super().__init__(model_files, actuation_spec, observation_spec, gamma,
                         horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, size=(500, 500), distance=1.8)

        self.pino_model = pino.buildModelFromUrdf(self.agents[0]['urdf'])
        se_tip = pino.SE3(np.eye(3), np.array([0., 0., 0.585]))
        self.pino_model.addBodyFrame('striker_rod_tip', 7, se_tip, self.pino_model.nframes - 1)
        self.pino_data = self.pino_model.createData()
        self.frame_idx = self.pino_model.nframes - 1

        self._client.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0.0, cameraPitch=-89.9,
                                                cameraTargetPosition=[0., 0., 0.])
        self.env_spec = dict()
        self.env_spec['table'] = {"length": 1.96, "width": 1.02, "height": 0.0, "goal": 0.25, "urdf": table_file}
        self.env_spec['puck'] = {"radius": 0.03165, "urdf": puck_file}
        self.env_spec['mallet'] = {"radius": 0.05}
        self.env_spec['universal_height'] = 0.1505

    def _compute_action(self, state, action):
        if self.step_action_function is None:
            ctrl_action = action
        else:
            ctrl_action = self.step_action_function(state, action)

        joint_state = self.joints.positions(state)[:9]

        if self.isolated_joint_7:
            joint_7_des_pos = self._compute_joint_7(joint_state)
            ctrl_action = np.concatenate([ctrl_action, joint_7_des_pos])

        joint_universal_pos = self._compute_universal_joint(joint_state)
        return np.concatenate([ctrl_action, joint_universal_pos])

    def _simulation_pre_step(self):
        if self.env_noise:
            force = np.concatenate([np.random.randn(2), [0]]) * 0.0005
            self._client.applyExternalForce(self._model_map['puck']['id'], -1, force, [0., 0., 0.],
                                            self._client.WORLD_FRAME)

    def is_absorbing(self, state):
        boundary = np.array([self.env_spec['table']['length'], self.env_spec['table']['width']]) / 2
        puck_pos = self.get_sim_state(state, "puck", PyBulletObservationType.BODY_POS)[:3]
        if np.any(np.abs(puck_pos[:2]) > boundary) or abs(puck_pos[2] - self.env_spec['table']['height']) > 0.1:
            return True

        boundary_mallet = boundary
        for agent in self.agents:
            mallet_pose = self.get_sim_state(state, agent['name'] + "/striker_mallet_tip",
                                             PyBulletObservationType.LINK_POS)
            if np.any(np.abs(mallet_pose[:2]) - boundary_mallet > 0.02):
                return True
        return False

    def _compute_joint_7(self, state):
        raise NotImplementedError

    def _compute_universal_joint(self, state):
        raise NotImplementedError
