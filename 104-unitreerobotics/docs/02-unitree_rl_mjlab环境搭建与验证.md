# unitree_rl_mjlab 环境搭建与验证（RTX 5090）

> 状态（2026-10-19）：环境搭建完成，§3 三判据验证通过；§6 阶段 1 smoke train 实测通过，阶段 2 起（play / 正式训练 / tracking）待验证。
> 仓库：`repos/unitree_rl_mjlab` — Unitree 官方 RL 运动控制库，基于 mjlab 1.2.0 插件式扩展，链路：训练（PPO / MuJoCo Warp）→ play 回放 → C++ 实机部署。
> 任务注册表实测 29 个：mjlab 内置 9（Yam / Go1 / 内置 G1）+ Unitree 插件 20（velocity 8 机型 × Flat/Rough，tracking G1、G1-23Dof × 有/无状态估计）。

## 1. 已验证版本组合

| 包 | 已验证版本 | 约束来源 / 选择理由 |
|---|---|---|
| Python | 3.11 | 官方文档指定；5090 兼容性取决于 torch 的 CUDA wheel（cu128+），与 Python 版本无关 |
| torch | 2.9.1+cu128 | sm_120 需 cu128+；同机 microduck_rl（mjlab 1.3.0）跑通的同根组合 |
| torchvision | 0.24.1 | 与 torch 2.9.1 官方严格配对（由 rsl-rl-lib 引入） |
| mjlab | 1.2.0 | setup.py 锁定 |
| mujoco-warp | 3.5.0 | setup.py 锁定 |
| mujoco | 3.5.0 | 3.5 系列仅此一版，与 mujoco-warp 3.5.0 同期互测；≥3.6 会崩（见 §4 坑 2） |
| warp-lang | 1.12.0 | 同时满足 mjlab（>=1.12.0）与 mujoco-warp（>=1.11.0）；本机 JIT 编译 5090 kernel 已验证 |
| tensordict | 0.10.0 | 支持 torch 2.9；最新 0.15.1 要求 torch>=2.13，会逼 pip 换掉 cu128 栈 |
| scipy | 最新即可 | mjlab 1.2.0 用了 scipy 但依赖声明漏了，需手动补装 |

## 2. 从零搭建命令（可复现）

前置：仓库已克隆到 `repos/unitree_rl_mjlab`；`104-unitreerobotics/constraints-unitree-rl.txt` 已就位（§2.1）。

```bash
cd ~/ws/104-unitreerobotics/repos/unitree_rl_mjlab

# 1) 建环境
conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab   # 未 conda init 时：source ~/miniconda3/bin/activate unitree_rl_mjlab

# 2) torch 栈锁 cu128（必须先于项目安装，防 PyPI 默认 wheel 覆盖）
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# 3) 预置三个易被 pip 爬新的包（原因见 §4）
pip install mujoco==3.5.0 tensordict==0.10.0 warp-lang==1.12.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4) 项目本体（带 constraints 双保险；此时不会再动上面已 pin 的包）
pip install -e . -c ../../constraints-unitree-rl.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5) 补装 mjlab 1.2.0 漏声明的 scipy
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.1 constraints-unitree-rl.txt（位于 104-unitreerobotics/ 根目录）

```text
# unitree_rl_mjlab 环境 pin（RTX 5090 / CUDA 12.8 已验证，2026-10-19）
torch==2.9.1
torchvision==0.24.1
mujoco==3.5.0
tensordict==0.10.0
warp-lang==1.12.0
```

以后向该环境追加任何包都建议 `pip install <pkg> -c ../../constraints-unitree-rl.txt`，防连带升级。

## 3. 验证判据（三绿）

```bash
# ① torch 栈
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 预期：2.9.1+cu128 12.8 True NVIDIA GeForce RTX 5090

# ② mjlab 仿真栈（实测：mujoco 3.5.0 | mujoco-warp 3.5.0 | warp 1.12.0 | tensordict 0.10.0 | mjlab 1.2.0）
python -c "import mujoco, mujoco_warp, mjlab; from importlib.metadata import version; print(mujoco.__version__, version('mujoco-warp'), version('warp-lang'), version('tensordict'), version('mjlab'))"

# ③ 任务注册表（必须在仓库根目录执行）
python scripts/list_envs.py
# 预期：29 个任务，Unitree-G1-Flat 在列
```

另跑 `pip check` 应输出 `No broken requirements found`。

## 4. 踩坑记录（为什么要处处显式 pin）

三个坑同一类根因：**pip 对宽松版本约束一律取最新**，而这个仓库的依赖声明恰好处处宽松/缺失。

1. **torch 被换成 cu130**：`pip install -e .` 触发 rsl-rl-lib 5.0.1 → `tensordict>=0.7.0`（无上限）→ pip 取最新 tensordict 0.15.1 → 它要求 `torch>=2.13` → torch 被升到 2.14.1+cu130（PyPI 默认 wheel），triton / cuda-toolkit / nvidia 全套被换成 cu13 系。
2. **mujoco 3.16 import 即崩**：mujoco-warp==3.5.0 对 mujoco 的约束 `>=3.4.0` 过宽 → pip 取最新 3.16.0 → mujoco-warp 3.5.0 引用的 `mjtEnableBit.mjENBL_MULTICCD` 已被新版 mujoco 删除，`AttributeError`。
3. **scipy 缺失**：mjlab 1.2.0 的 terrains 模块 `import scipy.interpolate`，但 requires_dist 未声明（1.3.0 已修），`ModuleNotFoundError: No module named 'scipy'`。

**结论**：该环境内任何 pip install 都显式带版本号；绝不裸跑 `pip install -e .`，重装一律加 `-c constraints-unitree-rl.txt`。

## 5. 孤儿包清理（可选，省约 3GB）

坏解析曾装入 cu13 系列，与在用的 `-cu12` 系列可共存（互不依赖，pip check 已确认无断裂），可择机卸载：

```bash
pip uninstall -y cuda-bindings cuda-pathfinder cuda-toolkit cwevals pyvers \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
  nvidia-nvjitlink nvidia-nvtx \
  nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvshmem-cu13
```

注意：无后缀的 `nvidia-*` 是 cu13 时代孤儿；带 `-cu12` 后缀的（torch 2.9.1 在用）**不能动**。

## 6. 训练与 play（待验证）

命令均须在 `repos/unitree_rl_mjlab` 目录下、conda 环境已激活时执行（`import src` 与 `logs/` 均依赖当前目录）。

```bash
# 阶段 1：smoke train（验证链路，预期几分钟；首次跑 mujoco-warp JIT 编译内核需等待）
python scripts/train.py Unitree-G1-Flat \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.logger tensorboard
# 判据：开头打印 device=cuda:0；reward 各项有值、无 NaN；
#       logs/rsl_rl/g1_velocity/<时间戳>/ 下有 model_*.pt + policy.onnx + params/{env,agent}.yaml
# ✅ 实测（2026-10-19，run: 2026-10-19_17-19-02）：全部通过。64 envs 约 4300 steps/s、单 iter 0.36s；
#    产物 model_0.pt + model_4.pt + policy.onnx + params/{agent,env}.yaml 齐全；
#    首次 JIT 编译约 30s（单核最长 10.3s，之后有 ~/.cache/warp 缓存不再重编）；
#    前几 iter fell_over 终止居多、Mean reward 为负属正常（策略尚未学会站立）

# 阶段 2：play 回放（无 DISPLAY 时用 viser，浏览器开 http://localhost:8080）
cd ~/ws/104-unitreerobotics/repos/unitree_rl_mjlab
python scripts/play.py Unitree-G1-Flat \
  --checkpoint-file logs/rsl_rl/g1_velocity/2026-10-19_17-19-02/model_4.pt \
  --viewer viser
# ✅ 实测（2026-10-19）：viser 页面正常打开、checkpoint 加载成功、场景可交互；
#    smoke 策略（5 iters）倒地属预期，仅验证链路
# 坑：--checkpoint-file 必须给完整相对路径（只给文件名会 FileNotFoundError）；play 无 --agent.* 参数组

# 阶段 3：正式训练（默认 10001 iters、每 100 iter 存 checkpoint+ONNX；另开终端 watch -n 5 nvidia-smi 盯显存，2048 起步）
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs 2048 --agent.logger tensorboard
# 中断续训：追加 --agent.resume True（自动取最新 checkpoint，写入新时间戳目录，不覆盖旧目录）
# 看曲线：tensorboard --logdir logs/rsl_rl --port 6006

# 阶段 4（可选）：G1 动作模仿（动作数据仓库自带）
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz --input-fps 30 --output-fps 50 --robot g1
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs 2048 --agent.logger tensorboard
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/dance1_subject2.npz \
  --checkpoint-file logs/rsl_rl/g1_tracking/<时间戳>/model_XX.pt --viewer viser
```

注：默认 logger 是 wandb（首次 init 触发云认证，无效 key 会在训练开始时报 401），验证阶段一律 `--agent.logger tensorboard` 绕开；想用 wandb 云监控时先 `wandb login --relogin` 并用 `python -c "import wandb; print(wandb.Api().viewer.entity)"` 端到端确认登录有效。
