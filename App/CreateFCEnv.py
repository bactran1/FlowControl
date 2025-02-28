import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Training for Flow Control.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#import math
import torch
import math
import os

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg#, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as mdp
# 
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

FC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/bactran/Documents/FlowControl/custom40ftStraight/custom40ftStraight.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={ "Roller_1_joint": 0.0, "Roller_2_joint": 0.0, "Roller_3_joint": 0.0, "Roller_4_joint": 0.0, "Roller_5_joint": 0.0,
                    "Roller_6_joint": 0.0, "Roller_7_joint": 0.0, "Roller_8_joint": 0.0, "Roller_9_joint": 0.0, "Roller_10_joint": 0.0,
                    "Roller_11_joint": 0.0, "Roller_12_joint": 0.0, "Roller_13_joint": 0.0, "Roller_14_joint": 0.0, "Roller_15_joint": 0.0,
                    "Roller_16_joint": 0.0, "Roller_17_joint": 0.0, "Roller_18_joint": 0.0, "Roller_19_joint": 0.0, "Roller_20_joint": 0.0,
                    "Roller_21_joint": 0.0, "Roller_22_joint": 0.0, "Roller_23_joint": 0.0, "Roller_24_joint": 0.0, "Roller_25_joint": 0.0,
                    "Roller_26_joint": 0.0, "Roller_27_joint": 0.0, "Roller_28_joint": 0.0, "Roller_29_joint": 0.0, "Roller_30_joint": 0.0,
                    "Roller_31_joint": 0.0, "Roller_32_joint": 0.0, "Roller_33_joint": 0.0, "Roller_34_joint": 0.0, "Roller_35_joint": 0.0,
                    "Roller_36_joint": 0.0, "Roller_37_joint": 0.0, "Roller_38_joint": 0.0, "Roller_39_joint": 0.0, "Roller_40_joint": 0.0,
                    "Roller_41_joint": 0.0, "Roller_42_joint": 0.0, "Roller_43_joint": 0.0, "Roller_44_joint": 0.0, "Roller_45_joint": 0.0,
                    "Roller_46_joint": 0.0, "Roller_47_joint": 0.0, "Roller_48_joint": 0.0, "Roller_49_joint": 0.0, "Roller_50_joint": 0.0,
                    "Roller_51_joint": 0.0, "Roller_52_joint": 0.0, "Roller_53_joint": 0.0, "Roller_54_joint": 0.0, "Roller_55_joint": 0.0,
                    "Roller_56_joint": 0.0, "Roller_57_joint": 0.0, "Roller_58_joint": 0.0, "Roller_59_joint": 0.0, "Roller_60_joint": 0.0,
                    "Roller_61_joint": 0.0, "Roller_62_joint": 0.0, "Roller_63_joint": 0.0, "Roller_64_joint": 0.0, "Roller_65_joint": 0.0,
                    "Roller_66_joint": 0.0, "Roller_67_joint": 0.0, "Roller_68_joint": 0.0, "Roller_69_joint": 0.0, "Roller_70_joint": 0.0,
                    "Roller_71_joint": 0.0, "Roller_72_joint": 0.0, "Roller_73_joint": 0.0, "Roller_74_joint": 0.0, "Roller_75_joint": 0.0,
                    "Roller_76_joint": 0.0, "Roller_77_joint": 0.0, "Roller_78_joint": 0.0, "Roller_79_joint": 0.0, "Roller_80_joint": 0.0,
                    "Roller_81_joint": 0.0, "Roller_82_joint": 0.0, "Roller_83_joint": 0.0, "Roller_84_joint": 0.0, "Roller_85_joint": 0.0,
                    "Roller_86_joint": 0.0, "Roller_87_joint": 0.0, "Roller_88_joint": 0.0, "Roller_89_joint": 0.0, "Roller_90_joint": 0.0,
                    "Roller_91_joint": 0.0, "Roller_92_joint": 0.0, "Roller_93_joint": 0.0, "Roller_94_joint": 0.0, "Roller_95_joint": 0.0,
                    "Roller_96_joint": 0.0, "Roller_97_joint": 0.0, "Roller_98_joint": 0.0, "Roller_99_joint": 0.0, "Roller_100_joint": 0.0,
                    "Roller_101_joint": 0.0, "Roller_102_joint": 0.0, "Roller_103_joint": 0.0, "Roller_104_joint": 0.0, "Roller_105_joint": 0.0,
                    "Roller_106_joint": 0.0, "Roller_107_joint": 0.0, "Roller_108_joint": 0.0, "Roller_109_joint": 0.0, "Roller_110_joint": 0.0,
                    "Roller_111_joint": 0.0, "Roller_112_joint": 0.0, "Roller_113_joint": 0.0, "Roller_114_joint": 0.0, "Roller_115_joint": 0.0,
                    "Roller_116_joint": 0.0, "Roller_117_joint": 0.0, "Roller_118_joint": 0.0, "Roller_119_joint": 0.0, "Roller_120_joint": 0.0,
                    "Roller_121_joint": 0.0, "Roller_122_joint": 0.0, "Roller_123_joint": 0.0, "Roller_124_joint": 0.0, "Roller_125_joint": 0.0,
                    "Roller_126_joint": 0.0, "Roller_127_joint": 0.0, "Roller_128_joint": 0.0}
    ),
    actuators={
        "joint_1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_1_joint","Roller_2_joint","Roller_3_joint","Roller_4_joint","Roller_5_joint",
                              "Roller_6_joint","Roller_7_joint","Roller_8_joint","Roller_9_joint","Roller_10_joint",
                              "Roller_11_joint","Roller_12_joint","Roller_13_joint","Roller_14_joint","Roller_15_joint",
                              "Roller_16_joint","Roller_17_joint","Roller_18_joint","Roller_19_joint","Roller_20_joint",
                              "Roller_21_joint","Roller_22_joint","Roller_23_joint","Roller_24_joint","Roller_25_joint",
                              "Roller_26_joint","Roller_27_joint","Roller_28_joint","Roller_29_joint","Roller_30_joint",
                              "Roller_31_joint","Roller_32_joint"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "joint_33_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_33_joint","Roller_34_joint","Roller_35_joint","Roller_36_joint","Roller_37_joint",
                              "Roller_38_joint","Roller_39_joint","Roller_40_joint","Roller_41_joint","Roller_42_joint",
                              "Roller_43_joint","Roller_44_joint","Roller_45_joint","Roller_46_joint","Roller_47_joint",
                              "Roller_48_joint","Roller_49_joint","Roller_50_joint","Roller_51_joint","Roller_52_joint",
                              "Roller_53_joint","Roller_54_joint","Roller_55_joint","Roller_56_joint","Roller_57_joint",
                              "Roller_58_joint","Roller_59_joint","Roller_60_joint","Roller_61_joint","Roller_62_joint",
                              "Roller_63_joint","Roller_64_joint"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "joint_65_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_65_joint","Roller_66_joint","Roller_67_joint","Roller_68_joint","Roller_69_joint",
                              "Roller_70_joint","Roller_71_joint","Roller_72_joint","Roller_73_joint","Roller_74_joint",
                              "Roller_75_joint","Roller_76_joint","Roller_77_joint","Roller_78_joint","Roller_79_joint",
                              "Roller_80_joint","Roller_81_joint","Roller_82_joint","Roller_83_joint","Roller_84_joint",
                              "Roller_85_joint","Roller_86_joint","Roller_87_joint","Roller_88_joint","Roller_89_joint",
                              "Roller_90_joint","Roller_91_joint","Roller_92_joint","Roller_93_joint","Roller_94_joint",
                              "Roller_95_joint","Roller_96_joint"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "joint_97_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_97_joint","Roller_98_joint","Roller_99_joint","Roller_100_joint","Roller_101_joint",
                              "Roller_102_joint","Roller_103_joint","Roller_104_joint","Roller_105_joint","Roller_106_joint",
                              "Roller_107_joint","Roller_108_joint","Roller_109_joint","Roller_110_joint","Roller_111_joint",
                              "Roller_112_joint","Roller_113_joint","Roller_114_joint","Roller_115_joint","Roller_116_joint",
                              "Roller_117_joint","Roller_118_joint","Roller_119_joint","Roller_120_joint","Roller_121_joint",
                              "Roller_122_joint","Roller_123_joint","Roller_124_joint","Roller_125_joint","Roller_126_joint",
                              "Roller_127_joint","Roller_128_joint"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        )
    },
)

@configclass
class FlowControlSceneCfg(InteractiveSceneCfg):
    """Configuration for a Flow Control Conveyor robot scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Flow Control section
    FC: ArticulationCfg = FC_CFG.replace(prim_path="/World/Loop")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_velocities = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["joint_1_actuator", "joint_33_actuator","joint_65_actuator","joint_97_actuator"], scale=1.0)
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: don't cross the torque limit
    limitTorque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("FC")},
        )
    # (4) Primary task: don't cross the speed limit
    limitTorque = RewTerm(
        func=mdp.joint_vel_l2,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("FC")},
        )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) The robot fell on the ground
    '''
    robot_on_the_ground = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.4},
    )
    '''
    #robot_on_the_ground = DoneTerm(
    #    func=mdp.bad_orientation,
    #    params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi/3},
    #)    

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()

@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass

@configclass
class FlowControlEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the wheeled quadruped environment."""

    # Scene settings
    scene: FlowControlSceneCfg = FlowControlSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.render_interval = self.decimation


def main():
    """Main function."""
    # create environment configuration
    print("Main")
    env_cfg = FlowControlSceneCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_vel = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_vel)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()