# unitree_rl_mjlab 环境搭建与验证（RTX 5090）

> 状态（2026-10-20）：全链路验证完成——环境三判据 ✓ / smoke train ✓ / play ✓ / velocity 正式训练 ✓（含塌陷诊断与最佳 checkpoint 定位）/ tracking 舞蹈模仿 ✓，实测数据均见 §6。C++ 实机部署（deploy/simulate）未涉及。
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

## 6. 训练与 play（全链路已验证）

命令均须在 `repos/unitree_rl_mjlab` 目录下、conda 环境已激活时执行（`import src` 与 `logs/` 均依赖当前目录）。

> 训练曲线怎么读（塌陷诊断、最佳 checkpoint 选择）与 reward / 终止条件的设计逻辑，见 [docs/03-unitree_rl_mjlab架构与关键技术解析.md](03-unitree_rl_mjlab架构与关键技术解析.md)。

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
# 中断续训：追加 --agent.resume True（自动取最新 checkpoint，写入新时间戳目录，不覆盖旧目录；max-iterations 是增量不是终点，详见阶段 5）
# 看曲线：tensorboard --logdir logs/rsl_rl --port 6006
cd ~/ws/104-unitreerobotics/repos/unitree_rl_mjlab
python scripts/play.py Unitree-G1-Flat \
  --checkpoint-file logs/rsl_rl/g1_velocity/2026-10-19_17-23-46/model_10000.pt \
  --viewer viser

# better（实测最佳 checkpoint：峰值在 iter 4965 而非末位）
python scripts/play.py Unitree-G1-Flat \
  --checkpoint-file logs/rsl_rl/g1_velocity/2026-10-19_17-23-46/model_5000.pt \
  --viewer viser
# ✅ 实测（2026-10-19，run: 2026-10-19_17-23-46）：跑满 10000 iters，94 分钟（0.56s/iter、约 87k steps/s）；
#    曲线诊断：mean_reward 峰值 39.4 @ iter 4965，iter 5000→6000 策略塌陷（reward -14、error_vel 恶化 75%）
#    后仅恢复至峰值七成——train cfg.curriculum = {} 全程无课程，属 PPO 长跑失稳（adaptive lr）；
#    最佳 checkpoint = model_5000（上面 better 命令），非末位 model_10000；
#    velocity 任务充分收敛点约 5000 iters，重跑可直接 --agent.max-iterations 5500 省一半时间；
#    play 现象：稳定行走不倒、在 ±0.5~1.0 随机指令内自主变速换向（play 无遥杆）

# 阶段 4：G1 动作模仿（动作数据仓库自带；csv_to_npz ✅ 已执行，1.3M CSV → 12M NPZ）
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz --input-fps 30 --output-fps 50 --robot g1

python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs 2048 --agent.logger tensorboard

# play 须同时给 --motion-file 与 --checkpoint-file（完整相对路径）
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/dance1_subject2.npz \
  --checkpoint-file logs/rsl_rl/g1_tracking/2026-10-19_20-01-54/model_29000.pt \
  --viewer viser
# ✅ 实测（2026-10-19 夜，run: 2026-10-19_20-01-54）：跑满 30000 iters（4.35h，0.52s/iter，61 个 checkpoint）；
#    健康单调爬升无塌陷：mean_reward -2.0 → 峰值 26.2 @ iter 29079 → 最佳 checkpoint = model_29000（非末位）；
#    Episode_Termination/ee_body_pos 400 → 0.29（末端跟踪超差终止基本消失）、episode 长度 462/500；
#    与 velocity 对比：tracking 带 motion 采样 curriculum，收敛慢但曲线稳，后半程仍有增益
#    坑：run 目录的 <时间戳>.onnx（7MB，内含 motion 数据）是末位 30000 权重导出——
#        C++ 部署要用峰值权重须从 model_29000.pt 重新导出 ONNX

# 阶段 5：中断续训 / 增量加训（2026-10-21 源码核实；以 rough 10000→13000 试探为例）
cd ~/ws/104-unitreerobotics/repos/unitree_rl_mjlab
python scripts/train.py Unitree-G1-Rough --env.scene.num-envs 2048 \
  --agent.max-iterations 3000 --agent.logger tensorboard \
  --agent.experiment-name g1_velocity_rough --agent.resume True
# 机制（三个隐式环节拼出「从 10000 开始」，源码级核实）：
#   ① resume=True 触发 get_checkpoint_path(log_path, run_dir=".*", checkpoint="model_.*.pt")：
#      两个默认正则 → 字母序最新 run 目录 → model_<iter> 补零 15 位后数值排序取最大 → model_10000.pt
#   ② runner.load() 把 checkpoint 内存的 iter 字段（=10000）写回 current_learning_iteration，
#      同时恢复 actor/critic 权重 + optimizer 状态 + 观测归一化统计量
#   ③ learn() 语义：start_it = current_learning_iteration；total_it = start_it + max_iterations →
#      --agent.max-iterations 是【增量】不是【终点】——想跑到 13000 要传 3000（约 65 分钟）；
#      误传 13000 会从 10000 跑到 23000（4.7 小时）
# 行为与判读：
#   resume 不写旧目录：在 <experiment>/ 下新建 <新时间戳>/ 存续训 checkpoint（首个即 model_10000.pt），
#   play 脚本「取最新 run」逻辑天然命中续训目录
#   curriculum 状态不随 checkpoint 保存，且新会话从头重放（2026-10-21 续训实测修正）：
#   terrain_levels 重爬（实测 3.5 随机初始化→按能力回落稳定 ~1.7）；command_vel 的课程计数器
#   env.common_step_counter 是会话内计数，resume 后归零 → stage0 小范围 (-0.5,1.0) 立即重新生效
#   （注意：模板默认 ranges 本身就是 (-1.0,2.0) 大范围，旧 run iter 5000 的「翻倍跳变」
#    实为 stage1 把 stage0 的临时小范围改回模板默认值，不是「升级」）
#   【试探结果】3000 iters 后 track_linear 0.245~0.263 与旧平台 0.25 逐位重叠、error_vel 维持
#   ~2.0（该指标 = episode 内逐步 ||cmd-actual|| 均值，量级指向 ~2 m/s 的指令无关速度习惯）
#   → 加 iter 无效实锤：平台是策略 attractor，要突破需改 reward 权重/指令课程配置重训
#   判读口径：只盯 Episode_Reward/track_linear_velocity 是否脱离 0.21~0.25 平台，别看总 reward
#   （rough 10000 iter 实测：该项在 iter 5000 指令范围翻倍处从 0.48 腰斩到 0.21 后 4000 iter 平台，
#    总 reward 的爬升来自 pose/foot_gait 等姿态项——续训试探就是为了验证这个平台是否真饱和）
```

注：默认 logger 是 wandb（首次 init 触发云认证，无效 key 会在训练开始时报 401），验证阶段一律 `--agent.logger tensorboard` 绕开；想用 wandb 云监控时先 `wandb login --relogin` 并用 `python -c "import wandb; print(wandb.Api().viewer.entity)"` 端到端确认登录有效。
