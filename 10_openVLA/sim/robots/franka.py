from __future__ import annotations

import numpy as np
from pathlib import Path

from .base import RobotConfig

FRANKA_CONFIG = RobotConfig(
    name="franka",
    robot_dir=Path("/home/creator/ws/assets/mujoco/mujoco_menagerie/franka_emika_panda"),
    actuator_names=[
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7",
        "actuator8",
    ],
    arm_joint_names=[
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ],
    home_ctrl=np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0], dtype=np.float64),
    gripper_body_name="hand",
    finger_link_marker='                      <body name="left_finger" pos="0 0 0.0584">',
    ee_site_pos=(0.0, 0.0, 0.108),
    gripper_open=255.0,
    gripper_closed=0.0,
    scene_xml_template="""<mujoco model="franka_openvla_scene">
  <option timestep="0.002" gravity="0 -9.81 0"/>
  <visual>
    <global azimuth="-135" elevation="-30" />
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.3 0.3 0.3" rgb2="0.4 0.4 0.4"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>
  </asset>
  <worldbody>
    <camera name="overview" pos="0.4 0.4 0.6" quat="0.886 0.123 0.272 0.354"/>
    <light name="ambient_light" pos="0 0 3" dir="0 0 -1" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" ambient="0.4 0.4 0.4"/>
    <light name="point_light_1" pos="0.5 0.5 1.0" dir="-0.5 -0.5 -1" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5"/>
    <light name="point_light_2" pos="-0.5 0.5 1.0" dir="0.5 -0.5 -1" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.02" rgba="0.7 0.7 0.7 1" material="grid"/>
    <body name="table" pos="{table_x} {table_y} 0">
      <geom name="table_top" type="box" size="0.35 0.35 0.02" rgba="0.5 0.4 0.3 1"/>
      <geom name="table_leg_1" type="box" size="0.02 0.02 0.48" pos="-0.33 0.33 -0.5" rgba="0.4 0.3 0.2 1"/>
      <geom name="table_leg_2" type="box" size="0.02 0.02 0.48" pos="-0.33 -0.33 -0.5" rgba="0.4 0.3 0.2 1"/>
      <geom name="table_leg_3" type="box" size="0.02 0.02 0.48" pos="0.33 0.33 -0.5" rgba="0.4 0.3 0.2 1"/>
      <geom name="table_leg_4" type="box" size="0.02 0.02 0.48" pos="0.33 -0.33 -0.5" rgba="0.4 0.3 0.2 1"/>
    </body>
    <body name="red_cube" pos="{cube_x} {cube_y} {cube_z}">
      <freejoint name="cube_freejoint"/>
      <geom name="cube_geom" type="box" size="{cube_size} {cube_size} {cube_size}" rgba="1.0 0.0 0.0 1.0" mass="0.1"/>
    </body>
    <include file="{robot_file}"/>
  </worldbody>
</mujoco>""",
)