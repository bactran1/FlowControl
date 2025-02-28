import argparse

from isaaclab.app import AppLauncher
from typing import List

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
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.utils import configclass

FC_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="C:/Users/bactran/Documents/FlowControl/custom40ftStraight.urdf",
        fix_base=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type="force", gains=1.0),
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
        joint_pos={ "Roller_*": 0.0,}
    ),
    actuators={
        "joint_1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_1_1_joint"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0
        ),
        "joint_33_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_33_33_joint"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0
        ),
        "joint_65_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_65_65_joint"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0
        ),
        "joint_97_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_97_97_joint"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0
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
    FC: ArticulationCfg = FC_CFG.replace(prim_path="{ENV_REGEX_NS}/Loop")

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

    joint_velocities = mdp.JointVelocityActionCfg(asset_name="FC", joint_names=["joint_1_actuator", "joint_33_actuator","joint_65_actuator","joint_97_actuator"], scale=1.0)
    

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
    env_cfg = FlowControlEnvCfg()
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
            print("[Env 0]:  Joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()