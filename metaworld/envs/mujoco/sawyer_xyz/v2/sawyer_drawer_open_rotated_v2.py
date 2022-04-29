import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.mujoco.utils.rotation import euler2quat, euler2mat

class SawyerDrawerOpenRotatedEnvV2(SawyerXYZEnv):
    def __init__(self, transparent_sawyer=False, 
            hand_near_drawer=False, hand_angle_delta=0, fingers_closed=True):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0, 0.0)
        obj_high = (0.1, 0.9, 0.0, 360.0)
        self._transparent_sawyer = transparent_sawyer
        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            transparent_sawyer=transparent_sawyer,
        )
        self.goal_delta = np.array([.0, -.16, .09])
        self.fingers_closed = fingers_closed
        self.hand_angle_delta = hand_angle_delta
        self.hand_near_drawer = hand_near_drawer
        self._angle = 0.
        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.0], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    def new_drawer_position(self, init_position, angle):
        init_p = init_position.copy()
        init_p[1] -= 0.1
        box_position = init_p.copy()
        # this is to align "switch" with train / test split
        # print(self.hand_angle_delta)
        self._angle = (int(angle) + 7 + self.hand_angle_delta) % 360 
        box_quat = euler2quat(np.array([0, 0, np.pi * angle / 180]))
        M = euler2mat(np.array([0, 0, np.pi * angle / 180]))
        # this is box_forth_edge - box_back_edge
        x = np.array([0., 0.27, 0])
        goal_delta = np.array([.0, -.16 - self.maxDist, .09])
        center = init_p.copy()[:3]
        center[1] -= x[1]
        center[2] = 0.09
        # center = np.array([box_position[0], box_position[1] - x[1], .09])
        box_position[:2] += (M @ x - x)[:2]
        # print(box_position)
        goal_position = M @ goal_delta + box_position
        # hand_delta = np.array([-.01, -.16 + 0.01, .06])
        if self.hand_near_drawer:
            hand_delta = np.array([0., -.16 + 0.023, 0.18])
            hand_position = M @ hand_delta + box_position
        else:
            # hand_position = self.hand_init_pos
            hand_position = center
            hand_position[2] = self.hand_init_pos[2]
        center = hand_position
        # normalize angle so the robot pose would be approx. the same for all angles.
        # [51; 53 + 180] -> [53; 53 + 180].
        # [53 + 180; 53 + 360] -> [53; 53 + 180].
        # 53 is is an angle of pose switch
        angle = (angle - 53 + 360 + self.hand_angle_delta) % 180 + 53
        hand_quat = euler2quat(np.array([np.pi/2, angle/180 * np.pi, -np.pi/2]))
        # print(hand_quat)
        return box_position, box_quat, goal_position, hand_position, hand_quat, center

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_drawer.xml',
            transparent_sawyer=self._transparent_sawyer)

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            gripper_error,
            gripped,
            handle_error,
            caging_reward,
            opening_reward
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(handle_error <= 0.03),
            'near_object': float(gripper_error <= 0.03),
            'grasp_success': float(gripped > 0),
            'grasp_reward': caging_reward,
            'in_place_reward': opening_reward,
            'obj_to_target': handle_error,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        if not hasattr(self, '_angle'):
            self._angle = 0.
        M = euler2mat(np.array([0, 0, np.pi * self._angle / 180]))
        delta = np.array([.0, -.16, .0])
        return self.get_body_com('drawer_link') + M @ delta

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('drawer_link')

    def reset_model(self):
        # self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        # Set mujoco body to computed position
        init_pos, angle = self.obj_init_pos[:3], self.obj_init_pos[-1] 
        box_position, box_quat, goal_position, hand_position, hand_quat, center = \
            self.new_drawer_position(init_pos, angle)
        # Set _target_pos to current drawer position (closed)
        self._target_pos = goal_position
        self.hand_init_pos = hand_position
        self.hand_init_quat = hand_quat
        self._reset_hand()
        m = 1 if self.fingers_closed else -1
        self.do_simulation([m, -m], 300)
        self.sim.model.body_pos[self.model.body_name2id('drawer')] = box_position
        self.sim.model.body_quat[self.model.body_name2id('drawer')] = box_quat
        self._set_obj_xyz(0)
        self._reset_hand()
        self.do_simulation([m, -m], 300)
        # self.data.set_mocap_pos('mocap', self.hand_init_pos)
        # self.data.set_mocap_quat('mocap', self.hand_init_quat)
        self._set_obj_xyz(0)
        self._reset_hand()
        self.do_simulation([m, -m], 300)
        # Set _target_pos to current drawer position (closed) minus an offset
        self._set_pos_site('goal', self._target_pos)
        return self._get_obs()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def compute_reward(self, action, obs):
        gripper = obs[:3]
        handle = obs[4:7]

        handle_error = np.linalg.norm(handle - self._target_pos)

        reward_for_opening = reward_utils.tolerance(
            handle_error,
            bounds=(0, 0.02),
            margin=self.maxDist,
            sigmoid='long_tail'
        )

        handle_pos_init = self._target_pos + np.array([.0, self.maxDist, .0])
        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward
        scale = np.array([3., 3., 1.])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - self.init_tcp) * scale

        reward_for_caging = reward_utils.tolerance(
            np.linalg.norm(gripper_error),
            bounds=(0, 0.01),
            margin=np.linalg.norm(gripper_error_init),
            sigmoid='long_tail'
        )

        reward = reward_for_caging + reward_for_opening
        reward *= 5.0

        return (
            reward,
            np.linalg.norm(handle - gripper),
            obs[3],
            handle_error,
            reward_for_caging,
            reward_for_opening
        )
