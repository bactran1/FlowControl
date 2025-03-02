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

import math
import torch
import math
import os
#import re

# For Camera
import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sensors import patterns
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import convert_dict_to_backend

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as mdp
# 
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.utils import configclass

def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    print("Test1")
    prim_utils.create_prim("/World/Origin_00", "Xform")
    print("Test2")
    prim_utils.create_prim("/World/Origin_01", "Xform")
    print("Test3")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    print("Test4")
    # Create camera
    camera = Camera(cfg=camera_cfg)
    print("Test5")

    return camera

FC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="custom40ftStraight.usd",
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
        joint_pos={ "Roller_1_.*": 0.0, "Roller_33_.*": 0.0, "Roller_65_.*": 0.0, "Roller_97_.*": 0.0,}
    ),
    actuators={
        "joint_1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_1_.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1.0
        ),
        "joint_33_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_33_.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1.0
        ),
        "joint_65_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_65_.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1.0
        ),
        "joint_97_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_97_.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1.0
        )
    },
)

@configclass
class FlowControlSceneCfg(InteractiveSceneCfg):
    """Configuration for a Flow Control Conveyor robot scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0))
    )

    # Flow Control section
    print("Before-----------------------------")
    robot: ArticulationCfg = FC_CFG.replace(prim_path="{ENV_REGEX_NS}/FC")
    #FC: ArticulationCfg = FC_CFG.replace(prim_path="/World/FC")
    print("After------------------------------")

    #sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FC/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane","instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )



    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
        #init_state=AssetBaseCfg.InitialStateCfg(pos= (0,0,50),rot=(0.738, 0.477, 0.477, 0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos= (0,0,50),rot=(0.0, 0.0, 0.0, 50.0)),
    )



@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_velocities = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["Roller_1_.*", "Roller_33_.*","Roller_65_.*","Roller_97_.*"], scale=1.0)
    

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
        params={"asset_cfg": SceneEntityCfg("robot")},
        )
    # (4) Primary task: don't cross the speed limit
    limitTorque = RewTerm(
        func=mdp.joint_vel_l2,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
        )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) The robot fell on the ground
    #'''
    #robot_on_the_ground = DoneTerm(
    #    func=mdp.root_height_below_minimum,
    #    params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.4},
    #)
    #'''
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
    scene: FlowControlSceneCfg = FlowControlSceneCfg(num_envs=4096, env_spacing=16.0)
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_vel = (
                scene["robot"].data.default_joint_vel.clone(),
            )
            #joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_vel
        # -- apply action to the robot
        scene["robot"].set_joint_velocity_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("Received id seg image: ", scene["camera"].data.output["instance_id_segmentation_fast"].shape)
        print("-------------------------------")

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
            print("[Env 0]: Joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

        # print information from the sensors
        #print("-------------------------------")
        #print(env["camera"])
        print(env)
        #print("Received shape of rgb   image: ", env["camera"].data.output["rgb"].shape)
        #print("Received shape of depth image: ", env["camera"].data.output["distance_to_image_plane"].shape)

    # close the environment
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()