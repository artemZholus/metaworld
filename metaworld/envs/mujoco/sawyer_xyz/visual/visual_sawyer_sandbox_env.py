import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_visual_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from .heuristics import (
    AlongBackWall,
    Ceiling,
    GreaterThanXValue,
    InFrontOf,
    LessThanXValue,
    LessThanYValue,
    OnTopOf,
)
from .tools import (
    Basketball,
    BasketballHoop,
    BinA,
    BinB,
    BinLid,
    ButtonBox,
    CoffeeMachine,
    CoffeeMug,
    Dial,
    Door,
    Drawer,
    ElectricalPlug,
    ElectricalOutlet,
    FaucetBase,
    HammerBody,
    Lever,
    NailBox,
    Puck,
    ScrewEye,
    ScrewEyePeg,
    Shelf,
    SoccerGoal,
    Thermos,
    ToasterHandle,
    Window,
)
from .tools.tool import get_position_of, set_position_of, get_vel_of
from .solver import Solver
from .voxelspace import VoxelSpace


class VisualSawyerSandboxEnv(SawyerXYZEnv):

    def __init__(self):

        liftThresh = 0.1
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0, 0.6, 0.02)
        obj_high = (0, 0.6, 0.02)
        goal_low = (-1., 0, 0.)
        goal_high = (1., 1., 1.)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0, 0.6, 0.02], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.4), dtype=np.float32),
        }
        self.goal = np.array([-0.2, 0.8, 0.05], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._all_tool_names = list(self.model.body_names)

        self.liftThresh = liftThresh
        self.max_path_length = 200

        goal_low = np.array(goal_low)
        goal_high = np.array(goal_high)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self._world = None
        self._solver = None

    @property
    def model_name(self):
        return full_visual_path_for('sawyer_xyz/sawyer_sandbox_small.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        self.curr_path_length += 1
        info = {
            'success': float(False)
        }

        self.show_bbox_for(Lever())

        return ob, 0, False, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        '''
        Note: At a later point it may be worth it to replace this with
        self._get_obj_pos_dict
        '''
        return self.data.site_xpos[self.model.site_name2id('RoundNut-8')]

    # def _get_obj_pos_dict(self):
    #     return {
    #         name: get_position_of(name) for name in self._obj_names
    #     }


    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.objHeight = self.data.site_xpos[self.model.site_name2id('RoundNut-8')][2]
        self.heightTarget = self.objHeight + self.liftThresh

        basketball = Basketball()
        hoop = BasketballHoop()
        bin_a = BinA()
        bin_b = BinB()
        bin_lid = BinLid()
        button = ButtonBox()
        coffee_machine = CoffeeMachine()
        coffee_mug = CoffeeMug()
        dial = Dial()
        door = Door()
        drawer = Drawer()
        plug = ElectricalPlug()
        outlet = ElectricalOutlet()
        faucet = FaucetBase()
        hammer = HammerBody()
        lever = Lever()
        nail = NailBox()
        puck = Puck()
        screw_eye = ScrewEye()
        screw_eye_peg = ScrewEyePeg()
        shelf = Shelf()
        soccer_goal = SoccerGoal()
        thermos = Thermos()
        toaster = ToasterHandle()
        window = Window()

        world = VoxelSpace((1.75, 0.7, 0.5), 100)
        solver = Solver(world)

        # Place large artifacts along the back of the table
        solver.apply(AlongBackWall(0.95 * world.size[1]), [
            door, nail, hoop, window, coffee_machine, drawer
        ], tries=2)

        # Place certain artifacts on top of one another to save space
        solver.apply(OnTopOf(coffee_machine),   [toaster])
        solver.apply(OnTopOf(drawer),           [shelf])
        solver.apply(OnTopOf(nail),             [outlet])
        solver.apply(OnTopOf(door),             [button])

        # Put the plug in the outlet
        plug.specified_pos = outlet.specified_pos + np.array([.044, .0, .131])
        solver.did_manual_set(plug)
        # Put the faucet under the basketball hoop
        faucet.specified_pos = hoop.specified_pos + np.array([.0, -.1, .0])
        faucet.specified_pos[2] = faucet.resting_pos_z * world.resolution
        solver.did_manual_set(faucet)

        # Place certain artifacts in front of one another to simplify config
        solver.apply(InFrontOf(window),         [soccer_goal])
        solver.apply(InFrontOf(nail),           [bin_a])
        solver.apply(InFrontOf(bin_a),          [bin_b])

        # The ceiling means that taller objects get placed along the edges
        # of the table. We place them first (note that `shuffle=False` so list
        # order matters) so that the shorter objects don't take up space along
        # the edges until tall objects have had a chance to fill that area.

        def ceiling(i, j):
            # A bowl-shaped ceiling, centered at the Sawyer
            i -= world.mat.shape[0] // 2
            return (0.02 * i * i) + (0.005 * j * j) + 20

        solver.apply(Ceiling(ceiling), [
            thermos, lever, screw_eye_peg, dial
        ], tries=20, shuffle=False)

        # At this point we only have a few objects left to place. They're all
        # tools (not artifacts) which means they can move around freely. As
        # such, they can ignore the immense bounding boxes of the the door and
        # drawer (NOTE: in the future, it may make sense to have separate
        # `bounding` and `clearance` boxes so that freejointed tools can
        # automatically ignore clearance boxes). For now, to ignore the
        # existing bounding boxes, we manually reset their voxels:
        world.fill_tool(door, value=False)
        world.fill_tool(drawer, value=False)

        edge_buffer = (world.size[0] - 1.0) / 2.0
        solver.apply([
            # Must have this Y constraint to avoid placing inside the door
            # and drawer whose voxels got erased
            LessThanYValue(0.75 * world.size[1]),
            GreaterThanXValue(edge_buffer),
            LessThanXValue(world.size[0] - edge_buffer)
        ], [
            hammer, basketball, screw_eye, bin_lid, coffee_mug, puck
        ], tries=50, shuffle=False)

        for tool in solver.tools:
            tool.specified_pos[0] -= world.size[0] / 2.0
            tool.specified_pos[1] += 0.3

        self._world = world
        self._solver = solver

        self._make_everything_match_solver()
        # if not self._check_config_stability():
        #     self.reset_model()

        # print(self.model.joint_names)
        # print(self.model.body_names)
        # print(self.model.jnt_pos)
        # print(self.model.jnt_type)

        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._target_pos)) + self.heightTarget
        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False
        self.placeCompleted = False

    def show_bbox_for(self, tool):
        tool_pos = get_position_of(tool, self.sim)
        for site, corner in zip(
                ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                tool.get_bbox_corners()
        ):
            self.sim.model.site_pos[
                self.model.site_name2id(f'BBox{site}')
            ] = tool_pos + np.array(corner)

    def _make_everything_match_solver(self):
        for tool in self._solver.tools:
            if tool.name not in self._all_tool_names:
                print(f'Skipping {tool.name} placement. You sure it\'s in XML?')
                continue
            if (tool.name + 'Joint' in self.model.joint_names):
                joint_name = tool.name + 'Joint'
                # qpos_idx = self.model.get_joint_qpos_addr(joint_name)
                # qpos_old = self.sim.data.qpos[qpos_idx[0]:qpos_idx[1]]
                qpos_old = self.sim.data.get_joint_qpos(joint_name)
                qpos_new = qpos_old.copy()

                print(tool.name)
                print(self.model.body_dofadr[self.model.body_name2id(tool.name)])

                qpos_new[:3] = tool.specified_pos
                qpos_new[3:] = np.round(qpos_old[3:], decimals=1)

                self.sim.data.set_joint_qpos(joint_name, qpos_new)
                self.sim.data.set_joint_qvel(joint_name, np.zeros(6))
            else:
                set_position_of(tool, self.sim, self.model)

    def _check_config_stability(self, steps=10):
        for _ in range(steps):
            self.sim.step()

            for tool in self._solver.tools:
                if tool.pos_is_static or tool.name not in self._all_tool_names:
                    continue
                vel = get_vel_of(tool, self.sim, self.model)
                if np.linalg.norm(vel) > 8:
                    return False

        return True