"""阶段⑥前置(M4-calib) · 用 LIBERO 官方环境离线标定一次, 固化初始布局常量到 JSON。

背景 (M4 首轮闭环诊断):
    自建场景闭环跑通但碗未抓起 —— EE 始终悬在 z 1.01~1.39, 从未下探到碗高度 (0.8)。
    根因是「③初始化侧」未对齐: 机器人基座位姿 / 臂初始 qpos / 碗盘世界坐标 /
    agentview 相机实际位姿, 都与我们手搭场景里的猜测值有偏差 (robosuite 会做
    基座-桌面相对摆放 + 相机补 base_xpos 偏移, 手工推导易错)。

策略 (保持闭环脚本不依赖 LIBERO):
    本脚本是**唯一**允许 import repos/LIBERO + robosuite 的标定工具:
      ① 起官方 libero_spatial 目标任务 env, reset + set_init_state(trial 0)
         + 10 步 dummy 沉降 (与 run_libero_eval 完全一致的复位链);
      ② 沉降后从底层 mujoco model/data 提取世界系常量:
         机器人基座 pos/quat、臂 7 关节 qpos、夹爪指 qpos、EE 位姿、
         碗/盘 body 位姿、agentview 相机 pos/quat、桌面 top 高度;
      ③ 固化到 sim/assets/calibration/libero_spatial_bowl_plate.json。
    之后 06_libero_scene_rollout.py / libero_spatial_scene.xml 只读该 JSON,
    闭环链路恢复「不依赖 repos/LIBERO 与 robosuite」。

运行期命名约定 (已从底层 model 的 valid names 实测确认):
    臂关节 robot0_joint1..7; 夹爪指 gripper0_finger_joint1/2;
    机器人基座 body robot0_base (注意 mount0_base / flat_stove_1_base 等同后缀干扰);
    末端控制 site gripper0_grip_site (OSC 实际跟踪的抓取中心); 末端 body robot0_right_hand;
    物体主 body *_main (akita_black_bowl_1_main / plate_1_main);
    桌面高度有现成 site table_top。

用法:
    cd /home/creator/ws/100-openVLA && source .venv/bin/activate
    MUJOCO_GL=egl python scripts/calibrate_libero_scene.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_REPO = PROJECT_ROOT / "repos" / "LIBERO"  # 包路径为 libero.libero, 注入仓库根
OUT_JSON = PROJECT_ROOT / "sim" / "assets" / "calibration" / "libero_spatial_bowl_plate.json"

TASK_NAME = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
TASK_SUITE = "libero_spatial"
TRIAL_IDX = 0  # 官方评测 episode 0 用的 init state
NUM_STEPS_WAIT = 10  # 与 run_libero_eval cfg.num_steps_wait 一致


def log_step(message: str):
    print(f"[CALIB] {message}", flush=True)


def install_mj_fullm_shim():
    """robosuite 1.4.1 × mujoco 3.x 兼容补丁 (标定脚本专用, 不改 site-packages)。

    根因: robosuite controllers/base_controller.py 的 update() 调
        mujoco.mj_fullM(model, mass_matrix, data.qM)   # mujoco 2.3 旧签名
    而 mujoco 3.x 绑定只收 mj_fullM(model, data, dst[nv,nv]) —— 第2个实参必须是
    MjData 对象, 无法用模块级代理适配 (拿不到 MjData 引用)。

    方案: 运行时取 update() 源码, 把那一行改成 mujoco 3.x 签名
        mujoco.mj_fullM(model, data._data, mass_matrix)
    在原模块命名空间里 exec 后替换 Controller.update。其余逻辑逐行保留。
    """
    import inspect

    import robosuite.controllers.base_controller as bc

    src = inspect.getsource(bc.Controller.update)
    old_line = "mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)"
    new_line = "mujoco.mj_fullM(self.sim.model._model, self.sim.data._data, mass_matrix)"
    if old_line not in src:
        if new_line in src:
            log_step("update() 已是 mujoco 3.x 签名, 无需补丁")
            return
        raise RuntimeError(f"Controller.update() 源码变动, 未找到待补丁行:\n{src}")
    src = src.replace(old_line, new_line)
    src = "\n".join(line[4:] if line.startswith("    ") else line for line in src.splitlines())
    exec(compile(src, "<Controller.update mujoco3 patch>", "exec"), bc.__dict__)
    bc.Controller.update = bc.__dict__["update"]
    log_step("已补丁 Controller.update 的 mj_fullM 调用 (robosuite 1.4.1 × mujoco 3.x)")


def main():
    sys.path.insert(0, str(LIBERO_REPO))
    sys.path.insert(0, str(PROJECT_ROOT))

    install_mj_fullm_shim()  # 需在 robosuite 环境创建前 (补丁 Controller.update)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401  (确认可导入)

    # 复用官方评测的建 env 函数 (resolution=256, 与 run_libero_eval 相同)。
    # libero_utils 内部用 experiments.robot.* 绝对导入, 故需注入 openvla 仓库根。
    openvla_repo = PROJECT_ROOT / "repos" / "openvla"
    sys.path.insert(0, str(openvla_repo / "experiments" / "robot" / "libero"))
    sys.path.insert(0, str(openvla_repo))
    from libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402

    # LIBERO 的 init 文件是 numpy 数组的 torch.load, torch>=2.6 默认 weights_only=True 会拦。
    # 文件来自本仓 repos/LIBERO 可信, 标定期间临时强制 weights_only=False。
    import torch

    _orig_torch_load = torch.load
    torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE]()

    task_id = None
    for tid in range(task_suite.n_tasks):
        if task_suite.get_task(tid).name == TASK_NAME:
            task_id = tid
            break
    if task_id is None:
        raise SystemExit(f"未在 {TASK_SUITE} 中找到任务: {TASK_NAME}")
    task = task_suite.get_task(task_id)
    log_step(f"任务: task_id={task_id} | language={task.language}")

    init_states = task_suite.get_task_init_states(task_id)
    torch.load = _orig_torch_load  # 恢复, 后续不受影响
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    log_step(f"env 就绪: {task_description} | init_states={init_states.shape}")

    # ---- 官方复位链: reset -> set_init_state -> 10 步 dummy 沉降 ----
    env.reset()
    obs = env.set_init_state(init_states[TRIAL_IDX])
    log_step("set_init_state 完成, 开始沉降...")
    for t in range(NUM_STEPS_WAIT + 1):
        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
    log_step(f"沉降 {NUM_STEPS_WAIT} 步完成 (沉降后位姿即模型 step0 看到的布局)")

    # ---- 提取底层 mujoco 世界系常量 ----
    base_env = env
    while not hasattr(base_env, "sim"):
        base_env = base_env.unwrapped
    sim = base_env.sim
    mujoco_model = sim.model._model
    mujoco_data = sim.data._data

    def body_state(name: str) -> dict:
        bid = mujoco_model.body(name).id
        return {
            "xpos": mujoco_data.xpos[bid].tolist(),
            "xquat(wxyz)": mujoco_data.xquat[bid].tolist(),
        }

    def qpos_of(joint_name: str) -> float:
        return float(mujoco_data.qpos[mujoco_model.joint(joint_name).qposadr[0]])

    def site_state(site_name: str) -> dict:
        sid = mujoco_model.site(site_name).id
        return {
            "name": site_name,
            "xpos": mujoco_data.site_xpos[sid].tolist(),
            "xmat(3x3 row-major)": mujoco_data.site_xmat[sid].reshape(3, 3).tolist(),
        }

    # 运行期真实命名见文件头 docstring; 全部用精确名, 避开同后缀干扰。
    arm_qpos = [qpos_of(f"robot0_joint{i}") for i in range(1, 8)]
    gripper_qpos = [qpos_of(f"gripper0_finger_joint{i}") for i in (1, 2)]
    ee_site = site_state("gripper0_grip_site")
    ee_pos = ee_site["xpos"]
    ee_body = {"name": "robot0_right_hand", **body_state("robot0_right_hand")}
    robot_base = {"name": "robot0_base", **body_state("robot0_base")}

    # agentview 相机 (LIBERO _setup_camera 加入, 世界系, 已含 base_xpos 补偿)
    cam_id = mujoco_model.cam("agentview").id
    agentview = {
        "pos": mujoco_model.cam_pos[cam_id].tolist(),
        "quat(wxyz)": mujoco_model.cam_quat[cam_id].tolist(),
    }

    # 桌面 top 世界高度: 直接用 robosuite arena 自带的 table_top site (比扫 geom 名稳健)
    table_top_sid = mujoco_model.site("table_top").id
    table_top_z = float(mujoco_data.site_xpos[table_top_sid][2])

    calib = {
        "source": "LIBERO official env (calibrate_libero_scene.py)",
        "task_suite": TASK_SUITE,
        "task_name": TASK_NAME,
        "task_id": task_id,
        "language_instruction": task_description,
        "trial_idx": TRIAL_IDX,
        "num_steps_wait": NUM_STEPS_WAIT,
        "robot_base": robot_base,
        "arm_qpos_after_settle": arm_qpos,
        "gripper_qpos_after_settle": gripper_qpos,
        "ee_site_after_settle": ee_site,
        "ee_pos_after_settle": ee_pos,
        "ee_body_after_settle": ee_body,
        "objects": {
            "akita_black_bowl": body_state("akita_black_bowl_1_main"),
            "plate": body_state("plate_1_main"),
        },
        "agentview_camera": agentview,
        "table_top_z_world": table_top_z,
        "note": (
            "所有位姿均为 LIBERO env 世界系, 沉降 10 control step 后提取; "
            "自建场景复刻时: robot base 按 robot_base 摆放, 臂/夹爪 qpos 写入 home "
            "keyframe, 碗/盘 freejoint qpos=[pos, quat(wxyz)], agentview 相机直接用 "
            "agentview_camera (已是世界系, 不要再补 base 偏移)。"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(calib, indent=2), encoding="utf-8")

    log_step(f"标定完成, 已固化 -> {OUT_JSON}")
    log_step(f"  robot_base        : {robot_base}")
    log_step(f"  arm_qpos (settle) : {[round(q, 4) for q in arm_qpos]}")
    log_step(f"  gripper_qpos      : {[round(q, 4) for q in gripper_qpos]}")
    log_step(f"  ee_pos (settle)   : {[round(v, 4) for v in ee_pos]}")
    log_step(f"  bowl              : {calib['objects']['akita_black_bowl']}")
    log_step(f"  plate             : {calib['objects']['plate']}")
    log_step(f"  agentview cam     : {agentview}")
    log_step(f"  table_top_z_world : {table_top_z}")


if __name__ == "__main__":
    main()
