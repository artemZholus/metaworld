import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerStickPushEnvV2(SawyerXYZEnv):
    def __init__(self):

        liftThresh = 0.04
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.08, 0.58, 0.000)
        obj_high = (-0.03, 0.62, 0.001)
        goal_low = (0.399, 0.55, 0.0199)
        goal_high = (0.401, 0.6, 0.0201)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'stick_init_pos': np.array([-0.1, 0.6, 0.02]),
            'hand_init_pos': np.array([0, 0.6, 0.2]),
        }
        self.goal = self.init_config['stick_init_pos']
        self.stick_init_pos = self.init_config['stick_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh

        # For now, fix the object initial position.
        self.obj_init_pos = np.array([0.2, 0.6, 0.0])
        self.obj_init_qpos = np.array([0.0, 0.0])
        self.obj_space = Box(np.array(obj_low), np.array(obj_high))
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_stick_obj.xml')

    @_assert_task_is_set
    def step(self, action):
        obs = super().step(action)
        stick = obs[4:7]
        container = obs[11:14]
        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place = self.compute_reward(action, obs)
        success = float(np.linalg.norm(container - self._target_pos) <= 0.12)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_object and (tcp_open > 0) and (stick[2] - 0.02 > self.obj_init_pos[2]))

        # print(np.linalg.norm(container - self._target_pos), success)

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,

        }
        self.curr_path_length += 1


        return obs, reward, False, info

    def _get_pos_objects(self):
        return np.hstack((
            self.get_body_com('stick').copy(),
            self._get_site_pos('insertion') + np.array([.0, .09, .0]),
        ))

    def _get_quat_objects(self):
        return np.hstack((
            Rotation.from_matrix(self.data.get_body_xmat('stick')).as_quat(),
            np.array([0.,0.,0.,0.])
        ))

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = self._get_site_pos(
            'insertion'
        ) + np.array([.0, .09, .0])
        return obs_dict

    def _set_stick_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[16:18] = pos.copy()
        qvel[16:18] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self.stick_init_pos = self.init_config['stick_init_pos']
        self._target_pos = np.array([0.4, 0.6, self.stick_init_pos[-1]])
        self.stickHeight = self.get_body_com('stick').copy()[2]
        self.heightTarget = self.stickHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.stick_init_pos = np.concatenate((goal_pos[:2], [self.stick_init_pos[-1]]))
            self._target_pos = np.concatenate((goal_pos[-3:-1], [self.stick_init_pos[-1]]))

        self._set_stick_xyz(self.stick_init_pos)
        self._set_obj_xyz(self.obj_init_qpos)
        self.obj_init_pos = self.get_body_com('object').copy()
        self.maxPlaceDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self.stick_init_pos)) + self.heightTarget
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - self._target_pos[:2])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.pickCompleted = False

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        stick = obs[4:7]
        container = obs[11:14]
        tcp_opened = obs[3]
        target = self._target_pos

        obj_to_target = np.linalg.norm(stick - target)
        tcp_to_obj = np.linalg.norm(stick - tcp)
        in_place_margin = (np.linalg.norm(self.obj_init_pos - target))

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        object_grasped = self._gripper_caging_reward(action=action,
                                                     obj_pos=stick,
                                                     obj_radius=0.015,
                                                     pad_success_margin=0.05,
                                                     object_reach_radius=0.01,
                                                     x_z_margin=0.01,
                                                     high_density=False)

        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.02 and (tcp_opened > 0) and (stick[2] - 0.01 > self.obj_init_pos[2]):
            reward += 1. + 5. * in_place
        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]
