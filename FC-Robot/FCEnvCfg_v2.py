import argparse

from isaaclab.app import AppLauncher
from typing import List

# add argparse arguments
#parser = argparse.ArgumentParser(description="Training for Flow Control.")
#parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
#AppLauncher.add_app_launcher_args(parser)
# parse the arguments
#args_cli = parser.parse_args()
#args_cli. enable_cameras=True

# launch omniverse app
#app_launcher = AppLauncher(args_cli)
#simulation_app = app_launcher.app

import math
import torch
import math
import os
#import re
from random import randrange, uniform

# For Camera
import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors import Camera, CameraCfg, TiledCameraCfg
from isaaclab.sensors import patterns
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import convert_dict_to_backend

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg
#import isaaclab.envs.mdp as mdp
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

#Feature Extractor - needed to import a py file
#from .feature_extractor import FeatureExtractor, FeatureExtractorCfg

#Occupancy Mapping - needed to import a py file
#from omni.isaac.occupancy_map.bindings import _occupancy_map


FC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #usd_path=os.environ['USERPROFILE'] + "custom40ftStraight_v2.usd",
        usd_path="C:/Users/bactran/Documents/IsaacLab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/FC-Robot/Custom40ftStraight_v2.usd",
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
        joint_pos={"Roller_1_.*": 0.0, "Roller_17_.*": 0.0, "Roller_33_.*": 0.0, "Roller_49_.*": 0.0, 
                   "Roller_65_.*": 0.0, "Roller_81_.*": 0.0, "Roller_97_.*": 0.0, "Roller_113_.*": 0.0}
    ),
    actuators={
        "joint_1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_1_.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1.0
        ),
        "joint_17_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_17_.*"],
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
        "joint_49_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_49_.*"],
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
        "joint_81_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_81_.*"],
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
        ),
        "joint_113_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Roller_113_.*"],
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
    
    # rigid objects
    BoxS1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BoxS1",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.5, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(uniform(-5, -0.5), uniform(-0.5,0.5), 1.5),ang_vel=(uniform(15,20), uniform(15,20), uniform(15,20)))
    )
    
    BoxS2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BoxS2",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.5, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(uniform(-5, -0.5), uniform(-0.5,0.5), 1.5),ang_vel=(uniform(15,20), uniform(15,20), uniform(15,20)))
    )
    
    BoxS3: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BoxS3",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.5, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(uniform(-5, -0.5), uniform(-0.5,0.5), 1.5),ang_vel=(uniform(15,20), uniform(15,20), uniform(15,20)))
    )
    
    BoxS4: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BoxS4",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.5, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(uniform(-5, -0.5), uniform(-0.5,0.5), 1.5),ang_vel=(uniform(15,20), uniform(15,20), uniform(15,20)))
    )
    
    BoxS5: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BoxS5",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.5, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(uniform(-5, -0.5), uniform(-0.5,0.5), 1.5),ang_vel=(uniform(15,20), uniform(15,20), uniform(15,20)))
    )
    
    BoxS6: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BoxS6",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.5, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(uniform(-5, -0.5), uniform(-0.5,0.5), 1.5),ang_vel=(uniform(15,20), uniform(15,20), uniform(15,20)))
    )
    
    BoxS7: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BoxS7",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.5, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(uniform(-5, -0.5), uniform(-0.5,0.5), 1.5),ang_vel=(uniform(15,20), uniform(15,20), uniform(15,20)))
    )
    
        
    # Flow Control section
    print("Before-----------------------------")
    robot: ArticulationCfg = FC_CFG.replace(prim_path="{ENV_REGEX_NS}/FC")
    #FC: ArticulationCfg = FC_CFG.replace(prim_path="/World/FC")
    print("After------------------------------")
    
    tiled_camera1: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera1",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.76, 0.0, 1.85), rot=(0.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=1.62, horizontal_aperture=1.8, vertical_aperture=1.5, clipping_range=(1.79, 1.80)
        ),
        width=100,
        height=100
    )
    
    tiled_camera2: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera2",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.76-1.52*1, 0.0, 1.9), rot=(0.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=1.62, horizontal_aperture=1.8, vertical_aperture=1.5, clipping_range=(1.79, 1.80)
        ),
        width=100,
        height=100
    )
    tiled_camera3: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera3",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.76-1.52*2, 0.0, 1.9), rot=(0.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=1.62, horizontal_aperture=1.8, vertical_aperture=1.5, clipping_range=(1.79, 1.80)
        ),
        width=100,
        height=100
    )
    tiled_camera4: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera4",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.76-1.52*3, 0.0, 1.9), rot=(0.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=1.62, horizontal_aperture=1.8, vertical_aperture=1.5, clipping_range=(1.79, 1.80)
        ),
        width=100,
        height=100
    )
    tiled_camera5: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera5",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.76-1.52*4, 0.0, 1.9), rot=(0.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=1.62, horizontal_aperture=1.8, vertical_aperture=1.5, clipping_range=(1.79, 1.80)
        ),
        width=100,
        height=100
    )
    tiled_camera6: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera6",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.76-1.52*5, 0.0, 1.9), rot=(0.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=1.62, horizontal_aperture=1.8, vertical_aperture=1.5, clipping_range=(1.79, 1.80)
        ),
        width=100,
        height=100
    )
    tiled_camera7: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera7",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.76-1.52*6, 0.0, 1.9), rot=(0.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=1.62, horizontal_aperture=1.8, vertical_aperture=1.5, clipping_range=(1.79, 1.80)
        ),
        width=100,
        height=100
    )
    tiled_camera8: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera8",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.76-1.52*7, 0.0, 1.9), rot=(0.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1, focus_distance=1.62, horizontal_aperture=1.8, vertical_aperture=1.5, clipping_range=(1.79, 1.80)
        ),
        width=100,
        height=100
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

    joint_velocities = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["Roller_1_.*", "Roller_17_.*", "Roller_33_.*", "Roller_49_.*", 
                                                                                   "Roller_65_.*", "Roller_81_.*", "Roller_97_.*", "Roller_113_.*"], scale=1.0)
    

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
class DepthObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        """Observations for policy group with depth images."""

        image1 = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera1"), "data_type": "distance_to_image_plane"}
        )
        image2 = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera2"), "data_type": "distance_to_image_plane"}
        )
        image3 = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera3"), "data_type": "distance_to_image_plane"}
        )
        image4 = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera4"), "data_type": "distance_to_image_plane"}
        )
        image5 = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera5"), "data_type": "distance_to_image_plane"}
        )
        image6 = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera6"), "data_type": "distance_to_image_plane"}
        )
        image7 = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera7"), "data_type": "distance_to_image_plane"}
        )
        image8 = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera8"), "data_type": "distance_to_image_plane"}
        )

    policy: ObsGroup = DepthCameraPolicyCfg()

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
    limitVel = RewTerm(
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
    #observations: ObservationsCfg = ObservationsCfg()
    observations: DepthObservationsCfg = DepthObservationsCfg()
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
        super().__post_init__()
        # general settings
        self.decimation = 2
        # simulation settings
        self.sim.dt = 0.05  # simulation timestep -> 200 Hz physics
        self.episode_length_s = 5

        #self.scene.ground = None
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 2.5)
        # simulation settings
        self.sim.render_interval = self.decimation


# def main():
#     """Main function."""
#     # create environment configuration
#     print("Main")
#     env_cfg = FlowControlEnvCfg()
#     env_cfg.scene.num_envs = args_cli.num_envs
#     # setup RL environment
#     env = ManagerBasedRLEnv(cfg=env_cfg)
    
    
    
    
    
#     # physx = omni.physx.acquire_physx_interface()
#     # stage_id = omni.usd.get_context().get_stage_id()

#     # generator = _occupancy_map.Generator(physx, stage_id)
#     # # 0.05m cell size, output buffer will have 4 for occupied cells, 5 for unoccupied, and 6 for cells that cannot be seen
#     # # this assumes your usd stage units are in m, and not cm
#     # generator.update_settings(.05, 4, 5, 6)
#     # # Set location to map from and the min and max bounds to map to
#     # generator.set_transform((1, 1, 0.15), (-1, -1, 0.15), (0.5, 0.5, 0.05))
#     # generator.generate2d()
#     # # Get locations of the occupied cells in the stage
#     # occupied_points = generator.get_occupied_positions()
#     # # Get locations of the unoccupied cells in the stage
#     # free_points = generator.get_free_positions()
#     # # Get computed 2d occupancy buffer
#     # buffer = generator.get_buffer()
#     # # Get dimensions for 2d buffer
#     # dims = generator.get_dimensions()
#     # print('occupied', occupied_points)
#     # print('free', free_points)
#     # # print('buffer', buffer)
#     # print('dims', dims)
    
    
    
#     import random

#     # simulate physics
#     count = 0
#     while simulation_app.is_running():
#         with torch.inference_mode():
            
#             # reset
#             if count % 300 == 0:
#                 count = 0
                
#                 # Work in Progress - randommize MTBHs positions every reset
#                 #print(env.scene.rigid_objects['BoxS1'].data.default_root_state)
#                 #BoxS1_rootState = env.scene.rigid_objects['BoxS1'].data.default_root_state.clone()
#                 #BoxS1_rootState[:,:3] = torch.tensor([uniform(-5, 0), uniform(-0.5,0.5), 1.0])
#                 #env.scene.rigid_objects['BoxS1'].write_root_pose_to_sim(BoxS1_rootState[:, :7])
#                 #env.scene.rigid_objects['BoxS1'].write_root_velocity_to_sim(BoxS1_rootState[:, :6])
#                 #env.scene.rigid_objects['BoxS1'].reset()
                
                
#                 env.reset()
#                 print("-" * 80)
#                 print("[INFO]: Resetting environment...")
                
#             env.scene.rigid_objects['BoxS1'].write_data_to_sim()
#             # sample random actions
#             joint_vel = torch.randn_like(env.action_manager.action)
#             # step the environment
#             obs, rew, terminated, truncated, info = env.step(joint_vel)
            
#             print(rew)
#             print(env.scene)
#             # print current orientation of pole
#             #print("[Env 0]: Joint: ", obs["policy"][0][1].item())
            
#             #env.scene.rigid_objects.update(0.05)
            
#             # update counter
#             count += 1
#             if count % 50 == 0:
#                 print(f"Root position (in world): {env.scene.rigid_objects['BoxS1'].data.root_state_w[:, :3]}")

#     # close the environment
#     env.close()

# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()