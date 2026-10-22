"""Assemble the G1 + Wuji Hand MuJoCo model (XML surgery on the vendored G1)."""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import mujoco

from sonic_config import FPS, G1_MESHES, G1_XML, SIM_SUBSTEPS, WUJI_DESCRIPTION, XIXI_SCENE

from sim_scenes import INTERACTIVE_SCENES, add_interactive_scene, resolve_scene
from sim_scenes.robot_cameras import add_robot_cameras



def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"{name!r} is missing from the MuJoCo model")
    return value


def add_xixi_scene(root: ET.Element, asset: ET.Element) -> None:
    if not XIXI_SCENE.exists():
        raise FileNotFoundError(f"Xixi scene not found: {XIXI_SCENE}")
    scene = ET.parse(XIXI_SCENE).getroot()
    for item in scene.find("asset"):
        item = copy.deepcopy(item)
        if item.tag == "mesh":
            item.set("file", str(XIXI_SCENE.parent / "assets" / item.get("file")))
        asset.append(item)
    for worldbody in root.findall("worldbody"):
        for item in list(worldbody):
            if item.tag == "geom" and item.get("name") == "floor":
                worldbody.remove(item)
    worldbody = root.find("worldbody")
    for item in scene.find("worldbody"):
        # The scene ships its own key/fill lights, which would stack on top of the
        # base scene's headlight and directional light and wash the render out.
        if item.tag == "light":
            continue
        worldbody.append(copy.deepcopy(item))


def build_model(xixi_scene: bool = False, *, scene: str | None = None) -> mujoco.MjModel:
    scene = resolve_scene(xixi_scene, scene)
    if not G1_XML.exists():
        raise FileNotFoundError(f"G1 model not found: {G1_XML}")
    if not WUJI_DESCRIPTION.exists():
        raise FileNotFoundError(f"Wuji Hand model not found: {WUJI_DESCRIPTION}")
    root = ET.parse(G1_XML).getroot()
    root.find("compiler").set("meshdir", str(G1_MESHES))
    asset = root.find("asset")
    for item in list(asset):
        if item.get("name", "").startswith(("left_hand_", "right_hand_")):
            asset.remove(item)
    ET.SubElement(
        asset, "mesh", name="wuji_g1_adapter",
        file=str(WUJI_DESCRIPTION / "hand/attachment/unitree-g1-attachment/unitree-g1-docking-adapter.stl"),
        scale="0.001 0.001 0.001",
    )
    for side in ("left", "right"):
        hand = ET.parse(WUJI_DESCRIPTION / f"hand/body/mjcf/{side}.xml").getroot()
        hand_meshes = WUJI_DESCRIPTION / f"hand/body/meshes/{side}"
        for mesh in hand.find("asset"):
            mesh = copy.deepcopy(mesh)
            mesh.set("file", str(hand_meshes / mesh.get("file")))
            asset.append(mesh)
        wrist = next(body for body in root.findall(".//body") if body.get("name") == f"{side}_wrist_yaw_link")
        for child in list(wrist):
            if ((child.tag == "body" and child.get("name", "").startswith(f"{side}_hand_"))
                    or (child.tag == "geom" and child.get("mesh", "").startswith(f"{side}_hand_"))):
                wrist.remove(child)
        ET.SubElement(
            wrist, "geom", type="mesh", mesh="wuji_g1_adapter", pos="0.0415 0 0",
            quat="0.70710678 0 0.70710678 0", rgba="0.2 0.2 0.2 1",
            contype="0", conaffinity="0",
        )
        palm = copy.deepcopy(hand.find("./worldbody/body"))
        palm.set("pos", "0.066 0 0")
        palm.set("quat", "0.70710678 0 0.70710678 0")
        wrist.append(palm)
    add_robot_cameras(root)
    if scene == "xixi":
        add_xixi_scene(root, asset)
    elif scene in INTERACTIVE_SCENES:
        add_interactive_scene(root, asset, scene)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = 1.0 / (FPS * SIM_SUBSTEPS)
    return model
