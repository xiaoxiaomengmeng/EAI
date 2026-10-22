"""Dex3 hand retargeting from the four glove key points."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bridge_config import ASSETS, GLOVE_TO_DEX3, HAND_ORIGIN, HAND_TASK
from bridge_source import load_glove_recording, resample_points



def build_hand_retargeter(side: str):
    try:
        from dex_retargeting.retargeting_config import RetargetingConfig
    except ImportError as error:
        raise RuntimeError("hand conversion requires: pip install dex-retargeting") from error

    names = [
        f"{side}_hand_{finger}_{joint}_joint"
        for finger, joint in (
            ("thumb", 0),
            ("thumb", 1),
            ("thumb", 2),
            ("index", 0),
            ("index", 1),
            ("middle", 0),
            ("middle", 1),
        )
    ]
    retargeter = RetargetingConfig(
        type="dexpilot",
        urdf_path=str(ASSETS / f"unitree_dex3_{side}.urdf"),
        target_joint_names=names,
        wrist_link_name="base_link",
        finger_tip_link_names=["thumb_tip", "index_tip", "middle_tip"],
        low_pass_alpha=0.2,
    ).build()
    order = [retargeter.joint_names.index(name) for name in names]
    return retargeter, order


def hand_references(points: np.ndarray, side: str) -> np.ndarray:
    points = np.einsum("ij,...kj->...ki", GLOVE_TO_DEX3[side], points)
    return points[..., HAND_TASK, :] - points[..., HAND_ORIGIN, :]


def retarget_hand(points: np.ndarray, side: str) -> np.ndarray:
    retargeter, order = build_hand_retargeter(side)
    # First solve: joint-limit midpoint. Later solves: prior frame result.
    return np.stack(
        [retargeter.retarget(ref)[order] for ref in hand_references(points, side)]
    ).astype(np.float32)


def load_hand_joints(body_path: Path, target: np.ndarray) -> dict[str, np.ndarray]:
    hands = {}
    for side in ("left", "right"):
        glove_path = body_path.with_name(
            body_path.name.replace("vr_full_body", f"glove_{side}")
        )
        if glove_path == body_path:
            raise ValueError("body recording filename must contain 'vr_full_body'")
        timestamps, points = load_glove_recording(glove_path)
        hands[f"{side}_hand_joints"] = retarget_hand(
            resample_points(timestamps, points, target), side
        )
    return hands
