"""M1 验证: 编译自建 libero_spatial 场景并渲染 agentview 一帧, 与 LIBERO 官方视角定性对比。

用法:
    cd /home/creator/ws/100-openVLA && source .venv/bin/activate
    MUJOCO_GL=egl python scripts/verify_libero_scene.py

产物:
    outputs/libero_scene_verify/agentview.png  (LIBERO 机位, 256x256, 与官方一致的分辨率)
    outputs/libero_scene_verify/agentview_hires.png (640x480 便于人眼检查)
    并打印场景关键诊断 (物体/末端位姿、碰撞对数)。
"""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

SCENE = Path(__file__).resolve().parents[1] / "sim" / "scenes" / "libero_spatial_scene.xml"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "libero_scene_verify"


def main():
    print(f"[M1] 编译场景: {SCENE}")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    # 应用 home keyframe (关节 home + 夹爪张开 + 物体初始位姿)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    print(f"[M1] 编译成功: nq={model.nq}, nv={model.nv}, nu={model.nu}, ngeom={model.ngeom}")

    # 关键位姿诊断
    grip = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")]
    bowl = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "akita_black_bowl_1")]
    plate = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate_1")]
    print(f"[M1] grip_site  = {np.round(grip, 4)}")
    print(f"[M1] bowl_1     = {np.round(bowl, 4)}")
    print(f"[M1] plate_1    = {np.round(plate, 4)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 与 LIBERO 一致的 256x256 agentview
    for tag, (w, h) in {"agentview": (256, 256), "agentview_hires": (640, 480)}.items():
        renderer = mujoco.Renderer(model, height=h, width=w)
        renderer.update_scene(data, camera="agentview")
        rgb = renderer.render()
        out = OUT_DIR / f"{tag}.png"
        Image.fromarray(rgb).save(out)
        print(f"[M1] 渲染 {tag} ({w}x{h}) -> {out}")
        renderer.close()

    print("[M1] 完成。请肉眼对比 outputs/libero_scene_verify/agentview_hires.png 与 LIBERO 官方 agentview。")


if __name__ == "__main__":
    main()
