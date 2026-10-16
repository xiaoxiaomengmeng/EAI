# LingBot-VLA 2.0 代码架构与运行逻辑详解

> 面向初次接触本项目的读者，先讲清「这是什么、代码怎么组织、一次训练/推理发生了什么」，再逐一回答 8 个源码级问题。所有结论均来自源码核实，关键处附文件与行号引用。

---

## 一、项目是什么

**LingBot-VLA 2.0** 是蚂蚁灵波（Robbyant）开源的**视觉-语言-动作（Vision-Language-Action, VLA）基础模型**，目标是把大规模预训练的成果真正落到「可靠可用的真机机器人」上。相比 1.0，它的三点核心升级：

1. **跨任务、跨本体泛化**：约 6 万小时预训练数据（5 万小时机器人轨迹，覆盖 20 种机器人本体 + 1 万小时第一视角人类视频）。
2. **统一动作空间**：用一个 **55 维**标准向量表达手臂、末端、夹爪、灵巧手、腰、头、移动底盘等异构信号，而不只是双臂操作。
3. **预测式动力学建模（双查询蒸馏）**：用「预测未来」作为代理任务，由 **DINO-Video** 提供语义时序先验、**LingBot-Depth / MoGe** 提供几何先验，蒸馏进模型。

它的技术范式可以一句话概括：

> **VLM 主干（Qwen3-VL）+ 动作专家（Qwen2 + MoE）双流逐层联合注意力，用 Flow Matching 生成连续动作块，训练时额外挂两个蒸馏头对齐「深度 + 未来视频」。**

底座借鉴了两个开源项目：**VeOmni**（字节的分布式训练框架，提供 FSDP2/EP/序列并行）与 **LeRobot**（HuggingFace 的机器人数据格式）。

- 训练入口：`bash train.sh tasks/vla/train_lingbotvla.py <config.yaml>`
- 部署入口：`python -m deploy.lingbot_vla_v2_policy --model_path <hf_ckpt> --port <p>`
- 环境：Python 3.12 + PyTorch 2.8 + flash-attn 2.8.3

---

## 关键词解读（速查）

**VLA（Vision-Language-Action）**：视觉-语言-动作模型，输入图像+语言指令，输出机器人动作，是本项目的模型类别。

**Flow Matching（流匹配）**：一种生成连续动作的方法，把「纯噪声」逐步去噪成「干净动作」，训练时学速度场、推理时做 Euler 积分（本项目 10 步，t 从 1→0）。

**Action Expert（动作专家）**：专门负责生成动作的子网络（本项目是 Qwen2 36 层），与负责理解观测的 VLM 主干分工协作。

**双流逐层联合注意力**：VLM 主干流与动作专家流每一层把各自的 Q/K/V 拼起来做一次统一注意力，让动作专家能读到视觉/语言信息。

**prefix / suffix**：prefix = 图像+语言+蒸馏 query（观测侧）；suffix = state + 带噪动作 + 时间步（动作侧）。两段拼成完整输入序列。

**统一动作表征（55 维）**：把不同机器人的异构信号（臂/末端/夹爪/灵巧手/腰/头/底盘）统一映射进一个 55 维标准向量，实现跨本体训练。

**MoE（Mixture of Experts，混合专家）**：动作专家每层的 MLP 换成「多个专家 + 路由」，每个 token 只激活 top-k 个专家，用相同算力扩容。

**Token 级路由**：每个 token 独立经 gate 打分选专家（本项目 32 选 4），而非整条序列共用。

**Shared Expert（共享专家）**：所有 token 都会额外过的一个公共专家，承载通用先验，与路由专家输出相加。

**Loss-free 负载均衡**：不加辅助 loss，靠给路由分数加一个「纠偏 bias」来防止专家旱涝不均/死专家（DeepSeek-V3 风格）。

**双查询蒸馏（Dual-Query Distillation）**：在 prefix 插两组空白 query token，分别用深度 teacher（MoGe）和未来视频 teacher（DINO-Video）监督，逼模型学会场景几何与未来演化。

**Teacher / Student（教师/学生蒸馏）**：teacher 是已训好的现成模型，产出目标；student（本模型）去对齐它，等于「免费」获得监督信号。

**Depth Query Token**：depth 在 prefix 里的载体，是可学习参数而非真实像素——这是它与 image token 的本质区别。

**KV Cache（键值缓存）**：推理时 prefix 只前向一次并缓存其 K/V，10 步去噪里只重算 suffix，大幅提速。

**Action Chunking（动作分块）**：模型一次产出一整段动作（如 50 步），机器人每拍取其中一步执行，用满 use_length 步后再重新推理。

**AdaRMS（AdaRMSNorm）**：把时间步作为 FiLM 条件（`(1+γ)·x̂+β`）调制动作专家每层的归一化，是 π₀ 所没有的时间注入方式。

**FlexAttention**：PyTorch 2.5+ 的可编程稀疏注意力，用 mask 函数表达任意注意力模式并编译成高效 kernel。

**BlockMask / flex_cached**：把 2D 语义 mask 压成块稀疏结构；V2 默认只在第 0 层构建一次、36 层复用（flex_cached）。

**prefix-LM mask**：用 `cumsum(att_masks)` 前缀和表达「prefix 双向/因果、suffix 动作内双向但看不到 prefix」的注意力可见性规则。

**FSDP2（Fully Sharded Data Parallel 2）**：把参数/梯度/优化器状态按层分片到多卡省显存，用时才 all-gather。

**EP（Expert Parallelism，专家并行）**：把 MoE 专家的权重按 `Shard(0)` 切到不同 GPU，配合 fused group-gemm 计算。

**Muon**：一种优化器，对矩阵型参数的梯度做 Newton-Schulz 正交化后更新；embedding/norm 等仍交给 AdamW。

**Newton-Schulz 迭代**：Muon 用来近似把梯度矩阵正交化的迭代算法（五次多项式迭代）。

**LeRobot**：HuggingFace 的机器人数据格式（parquet + mp4 + meta），本项目数据底座，兼容 v2/v3 API。

**FeatureTransform**：把原始 LeRobot 数据依据 `robot_configs/<name>.yaml` 映射进 55 维统一空间、归一化、生成 joint_mask 的核心变换（含 unapply 逆过程）。

**joint_mask**：标记 55 维里哪些关节维对当前本体有效，用于 loss 只统计真实维度。

**image_grid_thw**：Qwen3-VL 特有的 patch 网格 (t,h,w)，用于计算视觉 token 的位置编码。

**DCP（Distributed Checkpoint）**：PyTorch 分布式 checkpoint 格式，训练时保存、可转成 HF 格式供推理。

**RoboTwin 2.0**：双臂操作仿真基准（50 任务），本项目提供的后训练与闭环评测样例。

**VeOmni**：字节的分布式训练框架，提供 FSDP2/EP/序列并行能力，是本项目的训练底座之一。

---

## 二、代码架构

### 2.1 顶层目录职责

| 目录 | 职责 |
|------|------|
| `lingbotvla/models/` | 模型定义：VLA 主体、Qwen3-VL 主干、Qwen2 动作专家、MoE 层、视觉蒸馏 teacher（MoGe/DINO） |
| `lingbotvla/data/` | LeRobot 数据集加载、统一动作表征变换（`FeatureTransform`）、collator、动态 batching |
| `lingbotvla/distributed/` | FSDP2、序列并行（Ulysses）、MoE 专家并行（EP）、梯度裁剪 |
| `lingbotvla/optim/` | Muon / 分布式 Muon / AdamW、LR scheduler |
| `lingbotvla/checkpoint/` | DCP 分布式 checkpoint 与 HF 格式互转 |
| `tasks/vla/` | 训练脚本入口（`train_lingbotvla.py`、`train_pi0.py`） |
| `deploy/` | 真机/仿真部署，WebSocket 推理服务 |
| `scripts/` | 模型下载、归一化统计、开环评估 |
| `configs/` | 训练配置（`vla/`）与机器人特征映射（`robot_configs/`） |
| `experiment/robotwin/` | RoboTwin 2.0 仿真基准的部署与评测 |

### 2.2 模型三层嵌套结构

代码里有**两代实现并存**：`modeling_lingbot_vla.py`（V1，Qwen2.5-VL 主干）与 `modeling_lingbot_vla_v2.py`（V2，Qwen3-VL 主干）。V2 通过继承复用 V1 的 `FlowMatching`、`AdaRMSNorm` 等基础件。V2 的类层次是：

```
LingbotVlaV2Policy (PreTrainedModel)          ← 对外接口：forward 算 loss / sample_actions 出动作
  └── FlowMatchingV2 (nn.Module)               ← Flow Matching 逻辑 + 投影层 + 蒸馏头
        └── QwenvlWithExpertV2Model            ← 双流 Transformer，逐层联合注意力
              ├── qwenvl:     Qwen3-VL-4B       ← VLM 主干（视觉编码 + 语言），处理 prefix
              └── qwen_expert: Qwen2 (36 层)    ← 动作专家，处理 suffix（state + noisy action），含 MoE
```

对应源码：
- [`LingbotVlaV2Policy`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L1198)
- [`FlowMatchingV2`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L429)
- [`QwenvlWithExpertV2Model`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L121)

### 2.3 双流逐层联合注意力（最核心的设计）

[`QwenvlWithExpertV2Model.forward`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L311) 是整个模型的引擎。两个子模型**层数必须相等（都是 36 层）**，每一层做三件事：

1. 各流独立算出自己的 Q/K/V（`i==1` 是动作专家流，会额外注入时间条件 `ada_cond`）；
2. **把两流的 Q/K/V 沿序列维 `torch.cat` 起来，做一次统一注意力**（先 `apply_mrope` 加旋转位置编码，再走 FlexAttention）；
3. 按各自序列长度切回，分别过 MLP / MoE。

这样动作专家能「看到」VLM 的视觉/语言 token，而 prefix 通过 attention mask 看不到 suffix（因果隔离，见问题 8）。

### 2.4 一次训练 / 推理的数据流

```
训练:
  LeRobot 原始帧
    → FeatureTransform（映射到 55 维统一空间 + 归一化 + padding）
    → collator 打包 micro-batch
    → prefix(图 + 文 + 蒸馏 query token) & suffix(state + noisy action + time)
    → Qwen3-VL 流 ‖ Qwen2-MoE 专家流，逐层联合注意力
    → 预测速度场 v_t（Flow Matching）+ 两个蒸馏头（depth / future-video）
    → 多项 loss 求和 → Muon/FSDP2 优化

推理:
  WebSocket 收到观测
    → FeatureTransform.apply(policy_eval=True)
    → prefix 只前向一次并缓存 KV
    → 10 步 Euler 去噪(t: 1→0)，每步只重算 suffix
    → 反归一化 unapply → 返回动作 chunk
```

### 2.5 V1 与 V2 软件架构对比图

代码库里 V1、V2 **两代实现并存**：V1 在 `modeling_lingbot_vla.py`，V2 在 `modeling_lingbot_vla_v2.py`。V2 并非重写，而是**继承复用** V1 的 `FlowMatching`、`AdaRMSNorm`、`init_depth_heads`、`depth_emb_forward`、`video_emb_forward` 等基础件，只替换主干与注意力/路由等关键部位。

#### 分层结构对比

```
═══════════════════════════════════════ 多模态输入 · MULTIMODAL INPUT ═══════════════════════════════════════
  观测 prefix    │ images (多相机 RGB)  ·  img_masks (相机有效位)  ·  image_grid_thw (patch 网格, 仅 V2)
                 │ lang_tokens (指令文本)  ·  lang_masks (文本有效位)
  动作 suffix    │ state (55 维统一本体状态)  ·  actions (训练真值) / noise (推理高斯噪声)  ·  time (t ∈ [0,1])
  蒸馏 (训练)    │ pil_images (当前帧)  ·  future_pil_images (未来帧)  →  depth / video teacher 现场算 target
                 ▼  FeatureTransform：映射到 55 维统一空间 + 归一化 + padding
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
 ┌── V1 · LingbotVlaPolicy ──────────────────────────────────────────────────────────────────────────────────
 │  LingbotVlaPolicy (PreTrainedModel)  →  FlowMatching (nn.Module)  →  QwenvlWithExpertModel
 │   · projections : state_proj · action_in_proj · action_out_proj · action_time_mlp_in/out (time + action)
 │   · distill     : init_depth_heads · depth_emb_forward (深度蒸馏)
 │   · embed_prefix: 无 image_grid_thw、无 deepstack
 │   · VLM    = Qwen2.5-VL (hidden 2048, heads 16q / 2kv)      ·  attn = flex (每层重建 BlockMask)
 │   · Expert = Qwen2 36 层, dense MLP (MoE 可选 / 默认关)     ·  heads 16q / 2kv
 │   · dual-stream: 逐层 cat(QKV) → 统一注意力 → 切回
 └──────────────────────────────────────────────────────────────────────────────────────────────────────────
 ┌── V2 · LingbotVlaV2Policy   (FlowMatchingV2 extends FlowMatching, 继承 V1 内核) ────────────────────────────
 │  LingbotVlaV2Policy (PreTrainedModel)  →  FlowMatchingV2  →  QwenvlWithExpertV2Model
 │   · 复用 V1 全部投影层 + 蒸馏头 (init_depth_heads / video heads / AdaRMSNorm / Flow Matching 公式)
 │   · embed_prefix: + image_grid_thw、+ vision_start/end 边界 token、+ deepstack 多层视觉特征
 │   · position    : build_prefix_position_ids (get_rope_index / mrope) + _build_full_position_ids
 │   · VLM    = Qwen3-VL-4B (+ deepstack 视觉注入)             ·  attn = flex_cached (第 0 层建, 36 层复用)
 │   · Expert = Qwen2 36 层, MoE 全 36 层默认开 (32 专家 top-4 + shared expert)  ·  heads 32q / 8kv
 │   · dual-stream: 逐层 cat(QKV) → 统一注意力 → 切回 (机制同 V1)
 └──────────────────────────────────────────────────────────────────────────────────────────────────────────
                 │  action_out_proj：预测速度场 v_t → 10 步 Euler 去噪 (t: 1 → 0)
                 │  训练时另经 depth_align_head / video 头解码蒸馏 query token
                 ▼
════════════════════════════════════════════ 输出 · OUTPUT ════════════════════════════════════════════════════
  主输出         │ actions：55 维动作块 (chunk_size 步, 如 50)  →  unapply 反归一化  →  机器人真实关节空间
  训练辅助输出   │ depth_preds (当前深度) · future_depth_preds (未来深度) · future_video_preds (未来视频特征)
                 │ (仅训练期存在, 与 teacher 算蒸馏 loss；推理不需 teacher, 也不产出这些)
══════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

#### 关键差异一览（一关键词一行）

**VLM 主干**：V1 = Qwen2.5-VL（hidden 2048）；V2 = Qwen3-VL-4B（更强底座，带 deepstack 多层视觉特征注入）。

**embed_prefix 签名**：V1 = `(images, img_masks, lang_tokens, lang_masks, vlm_causal, precompute_grid_thw)`；V2 额外**必须**传 `image_grid_thw`（Qwen3-VL 处理器产出的 patch 网格）。

**视觉边界 token**：V1 默认 `qwen3vl_use_vision_boundaries=False`；V2 默认 True，图像块前后包 `vision_start_token` / `vision_end_token`。

**注意力实现**：V1 默认 `flex`（每层 `create_mask` 重建 BlockMask）；V2 默认 `flex_cached`（第 0 层建一次、36 层复用，省显存分配）。

**动作专家 MLP**：V1 稠密 MLP（MoE 仅可选开关）；V2 MoE **默认全 36 层开启**（robotwin：32 专家 top-4 + shared expert）。

**动作专家注意力头**：V1 = 16 query / 2 kv；V2 = 32 query / 8 kv（GQA）。

**位置编码**：V1 走 V1 内部逻辑；V2 用 `get_rope_index` 算 mrope 的 `build_prefix_position_ids` + `_build_full_position_ids` 拼接 prefix/suffix。

**deepstack**：V1 无；V2 有 `_apply_deepstack`，把 Qwen3-VL 的多层视觉特征逐层注入主干。

**Config 类**：V1 = `LingbotVLAConfig`（vocab 默认按 paligemma/qwen 推断）；V2 = `LingbotVLAV2Config`（`vlm_family="qwen3_vl"`，默认打开 chat_template / grid_thw / fixed_grid_cache）。

**继承复用（两代相同）**：Flow Matching 公式、`AdaRMSNorm` 时间调制、深度/未来视频蒸馏头、双流联合注意力机制、`make_att_2d_masks` 语义 mask——这些 V2 直接沿用 V1 代码，未重写。

> **一句话理解**：V2 = V1 的「换芯升级版」——把 VLM 主干从 Qwen2.5-VL 换成 Qwen3-VL-4B（带来 deepstack、视觉边界、grid_thw、mrope），动作专家从稠密 MLP 升级为全层 MoE、注意力头翻倍，注意力从 flex 换成 flex_cached 复用；而 Flow Matching、蒸馏、双流注意力等算法内核完全继承自 V1。

---

## 三、八个问题逐一详解

### 问题 1：`deploy/lingbot_vla_v2_policy.py` 里的 policy 怎么理解

**关键区分：这里有两层「policy」，别混淆。**

#### (1) 模型层的 policy —— `LingbotVlaV2Policy`
它是 [`nn.Module`/`PreTrainedModel`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L1198)，就是神经网络本身。在强化学习语义里，「policy（策略）」= 给定观测输出动作的映射 `π(a|o)`。这个类的两个方法正好对应策略的两种用法：
- `forward(...)`：训练用，输入含真值 `actions`，输出 loss；
- `sample_actions(...)`：推理用，输入观测，**采样**出一段动作块（Flow Matching 去噪）。

#### (2) 部署层的 policy 封装 —— `LingbotVLAv2Server`
[`deploy/lingbot_vla_v2_policy.py`](../lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py) 把裸模型包装成**能被机器人/仿真调用的服务**，它才是部署语境下的「policy 对象」。里面有三个角色：

| 类 | 作用 |
|----|------|
| [`PolicyPreprocessMixin`](../lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py#L56) | 提供 `select_action` / `sample_actions_batch`：把观测搬上 GPU、选 dtype、调 `sample_actions`、反归一化 |
| [`LingBotVlaV2InferencePolicy`](../lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py#L180) | `class ...(PolicyPreprocessMixin, LingbotVlaV2Policy): pass` —— 用多继承把「预处理能力」混入「模型本体」 |
| [`LingbotVLAv2Server`](../lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py#L184) | 真正的部署封装：加载权重、管理动作块缓存、对外暴露 `infer()` |

`LingbotVLAv2Server` 做的三件核心事：

**① 加载与初始化**（[`load_vla`](../lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py#L272)）：从 `hf_ckpt` 的上层目录读回训练时的 `lingbotvla_cli.yaml`，据此重建 `LingbotVLAV2Config`；把推理 attention 切成 `eager`（比 flex 更快）、`use_cache=True`（KV 缓存必需）；合并 Qwen3-VL 的 text/vision config；用 `safetensors` 加载权重（`strict=True`）。可选 `torch.compile` 加速。

**② 动作块缓存与分块执行**（[`infer`](../lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py#L461)）：这是部署 policy 的精髓。模型一次前向产出的是**一整段动作 chunk（如 50 步）**，但机器人每拍只需要一个动作。`infer` 用 `should_forward` 判断是否要重新推理：

```python
should_forward = (
    self.chunk_ret                       # 直接整块返回给上层
    or self.last_action_chunk is None    # 首次
    or self.global_step % self.use_length == 0   # 缓存用完了，该刷新
    or self.use_length == -1
)
```

- `chunk_ret=True`：整块返回，由上层（仿真/机器人）自己消费；
- `chunk_ret=False`：从缓存的 chunk 里按 `step_idx = global_step % use_length` 取**当前这一步**动作返回，`use_length` 步后才重新前向。这就是 VLA 常见的 **action chunking（动作分块）** 执行策略——减少推理次数、平滑动作。

**③ 特征变换**（[`_prepare_model_input`](../lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py#L401) / [`_unapply_batched_actions`](../lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py#L374)）：进来时 `FeatureTransform.apply` 把原始观测编码成模型输入；出去时 `unapply` 把归一化的动作还原回机器人真实关节空间。`reset()` 时会根据 `robo_name` 重新加载对应机器人的 `robot_configs/<name>.yaml` 和归一化统计。

> **一句话理解**：模型层 policy = 神经网络 `π(a|o)`；部署层 policy = 「权重加载 + 预处理 + 动作分块缓存 + WebSocket 服务」的运行时封装，让模型能被真实机器人以「每拍一个动作」的方式调用。

---

### 问题 2：`embed_prefix`（499-506）如何处理 depth，它和 image 有什么区别

先厘清一个**重要事实**：`embed_prefix` 本身**不计算深度**，深度只以「**查询 token（query token）占位符**」的形式被拼进 prefix 序列。真正的深度预测发生在模型 forward 之后的蒸馏头里。

#### (1) image 在 prefix 里是什么
[`embed_prefix`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L499) 里，图像走完整的视觉编码：
```python
img_emb, deepstack_embs = self.qwenvl_with_expert.embed_image(images, flat_grid_thw)
```
即经 Qwen3-VL 的 ViT 编码成 patch token，再包上 `vision_start_token` / `vision_end_token` 边界（`qwen3vl_use_vision_boundaries=True`）。**image 携带真实像素信息**，是模型的核心观测输入。

#### (2) depth 在 prefix 里是什么
depth 不是像素，而是一组**可学习的空白查询向量**，作为「占位符」插入序列。当 `use_depth_align and align_type=="query"` 时（[`embed_prefix` 的 depth-query 分支](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L582)）：

```python
current_task = _get_align_tokens(self.depth_align_embs)   # 可学习参数
align_embs   = current_task.repeat(...)                    # 广播到 batch
parts = [img_emb]                                          # 图像块打头
# 之后按 prefix_query_segments 顺序追加：language / current_depth / future_depth / future_video ...
```

`self.depth_align_embs` 是 [`init_depth_heads`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py#L799) 里注册的 `nn.Parameter`（形状 `[num_backbone_tokens, dim_out]`，如 256×2560）。`_get_align_tokens` 把它们 reshape 成 `num_task_tokens`（如 8）个 token。这些 token **初始时不含任何深度信息**，它们只是「预留座位」——在双流注意力里，它们能读到 image/lang token，于是被「训练成」承载场景几何的表征。

段的排列顺序由 [`prefix_query_segments`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/utils.py#L97) 决定：`language → current_depth → (future_video_cls) → (future_video) → future_depth`，future_depth 永远放最后（便于后续 mask 阻断）。

#### (3) image 与 depth 的本质区别

| 维度 | image token | depth（query）token |
|------|-------------|---------------------|
| 来源 | 真实像素经 ViT 编码 | 可学习参数，无输入信息 |
| 角色 | 观测输入（模型的「眼睛」） | 预测输出的「占位/探针」 |
| 数量 | 由 `image_grid_thw` 决定（patch 数 + 2 边界） | 固定 `num_task_tokens`（如 8） |
| 位置 | prefix 最前 | image + language 之后 |
| 训练信号 | 无直接监督（靠动作 loss 反传） | 有**深度 teacher 蒸馏 loss** 直接监督 |

#### (4) depth 到底在哪算、怎么监督
在 [`FlowMatchingV2.forward`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L846) 拿到主干输出 `outputs_embeds` 后，调 [`depth_emb_forward`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py#L1418)：
```python
align_embs = self._current_depth_task_tokens(hidden_states, ...)  # 取出 current_depth 那几个 token
align_embs = torch.cat([image_embs, align_embs], dim=1)
depth_preds = self.depth_align_head(align_embs, depth_preds)      # 深度头解码成深度图
current_loss = self._emb_loss(depth_preds, depth_targets)         # 与 teacher 深度对齐
```
`depth_targets` 是训练循环里用 **MoGe / MoRGBD** teacher 现场算出来的（`get_depth_target`，见 `train_lingbotvla.py` 769 行）。所以 depth query token 的作用是：**把「预测深度」这个几何理解任务，转成序列里的一个位置，让主干在因果推理时被迫学会场景 3D 结构**。

> **一句话理解**：image 是「输入的真实像素」，depth 是「插在序列里的空白探针 + 事后用深度头解码 + teacher 蒸馏监督」。前者是观测，后者是自监督代理任务的载体。

---

### 问题 3：793-794 的 Flow Matching 和 π₀ 的有什么区别

**核心结论：Flow Matching 的数学内核两者完全相同**，差异全在内核之外。经源码核实，同仓库 `models/vla/pi0/modeling_pi0.py` 与 `modeling_lingbot_vla_v2.py` 的这两行逐字一致：

```python
# lingbot-vla-v2 (L793-794) 与 pi0 (L2082-2083) 完全相同
time_expanded = time[:, None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions   # 噪声与动作的线性插值
u_t = noise - actions                                          # 目标速度场（常数）
```

而且连**时间采样**和**积分方向**都一样：
- `sample_time` 都用 `sample_beta(1.5, 1.0)` 再 `*0.999+0.001`（V1 [`sample_time`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py#L1055)、pi0 1935 行）；
- 采样都从 `t=1`（纯噪声）向 `t=0`（干净动作）做 Euler 积分，`dt = -1/num_steps`，默认 10 步（[sample_actions](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L971)、pi0 2151 行）。

这就是「原理相似」的原因——它们是**同一套 Rectified-Flow / Flow-Matching 动作生成范式**。

#### 那区别在哪？（全在内核之外）

| 维度 | π₀ | LingBot-VLA 2.0 |
|------|-----|------------------|
| **VLM 主干** | PaliGemma / Gemma 系 | **Qwen3-VL-4B**（更强的中文/视觉语言底座） |
| **动作专家** | 稠密 Gemma expert | **Qwen2 + 稀疏 MoE**（32 专家 top-4，跨本体扩展） |
| **动作空间** | 双臂操作为主（~14 维） | **统一 55 维**（臂/末端/夹爪/灵巧手/腰/头/底盘） |
| **时间步注入** | 时间 emb 拼进 action token（MLP 融合） | 额外支持 **AdaRMS**：时间作为 FiLM 条件调制专家每层归一化（`adanorm_time=True`） |
| **辅助任务** | 无 | **双查询蒸馏**：depth + future-video 两个代理 loss |
| **注意力实现** | flex | **flex_cached**（预建 BlockMask 跨层复用） |

其中「时间步注入」值得展开：π₀ 和 LingBot 都在 [`embed_suffix`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py#L1126) 里把正弦时间编码与 noisy action 拼接后过 MLP（`action_time_mlp_in/out`）。但 LingBot **额外**通过 `ada_cond=time_embs` 把时间送进动作专家每层的 `AdaRMSNorm`（[QwenvlWithExpertV2Model.forward 342 行](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L342)），即 DiT 风格的 FiLM 调制 `(1+γ)·x̂+β`。π₀ 无此机制。

> **一句话理解**：把 Flow Matching 想成「同一款发动机」，π₀ 和 LingBot-VLA 2.0 用的是同款发动机；区别在于 LingBot 换了更强的车头（Qwen3-VL）、加了多档变速箱（MoE 专家）、统一了轮距（55 维动作）、还装了两套辅助训练轮（深度 + 未来视频蒸馏）。

---

### 问题 4：MoE 动作专家如何路由

MoE 只装在**动作专家（qwen_expert）**里，不在 VLM 主干里。[`_install_moe_blocks`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L170) 把 `token_moe_layers` 指定的层（robotwin 配置是全部 36 层）的 MLP 替换成 [`Qwen2TokenMoeBlock`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py#L222)。

#### (1) 路由五步（`Qwen2TokenMoeBlock.forward`，L274）
```python
hidden_flat = hidden_states.reshape(-1, hidden_dim)          # (B*T, D) 逐 token
# ① gate 打分，强制 fp32（bf16 会让近似分数翻转 top-k → 路由抖动）
router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())
# ② 激活：sigmoid（robotwin）或 softmax
routing_scores = router_logits.sigmoid()   # or F.softmax(...)
# ③ 加「纠偏 bias」后选 top-k
scores_for_choice = routing_scores + self.e_score_correction_bias
_, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1)   # top_k=4
# ④ 取被选中专家的真实分数并归一化
routing_weights = routing_scores.gather(1, selected_experts)
if self.norm_topk_prob:
    routing_weights = routing_weights / (routing_weights.sum(-1, keepdim=True) + 1e-20)
routing_weights = routing_weights * self.routed_scaling_factor    # ×4.0
# ⑤ 专家计算（fused group-gemm 或 eager 逐专家）+ 加权求和
```
关键细节：
- **token 级路由**：每个 token 独立选专家（不是每序列），32 选 4；
- **gate 用 fp32**：注释明确说 bf16 logits 会在分数接近时翻转 top-k，造成「轮换死专家」；
- **打分与选择解耦**：`sigmoid` 分数既用于选择（加 bias 后 topk），也用于加权（用未加 bias 的原分数 gather）；
- **共享专家（shared expert）**：[`Qwen2TokenMoeBlock.forward`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py#L356) 所有 token 都过一个 `shared_expert`，与路由专家输出相加——这就是 README 说的「shared expert isolation」，让通用先验和特化模式共存。

#### (2) 负载均衡：两种机制并存
MoE 的老大难是「专家旱涝不均/死专家」。本项目提供两条路：

**A. Loss-free 均衡（无辅助 loss，靠 bias 纠偏）** —— [`moe_load_balance.py`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/moe_load_balance.py)
注册成 optimizer 的 pre-hook，每次 `optimizer.step()` 前执行（DeepSeek-V3 风格）：
```python
mean_load = tpe.float().mean()
deviation = (tpe.float() - mean_load).sign()
block.e_score_correction_bias.add_(-coeff * deviation)   # 过载专家 bias↓，欠载↑
```
`coeff = bias_update_speed`。robotwin 后训练里 `bias_update_speed: 0`（即关闭 loss-free，走下面的 B）。

**B. 辅助 loss 均衡** —— [`_moe_losses_and_metrics`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L1073)
- **序列级均衡 loss**（`sequence_wise_loss_coeff=1e-3`，`mode=per_sequence`）：在每个样本自己的 T 个 token 内平衡专家使用（Triton kernel `triton_sequence_wise_balance_loss`）；
- **router z-loss**（`router_z_loss_coeff=1e-4`）：`logsumexp(logits)²` 抑制 logits 爆炸。

#### (3) 专家并行（EP）
[`get_parallel_plan`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L1204) 把 `experts.gate_proj/up_proj/down_proj` 按 `Shard(0)` 切到不同 GPU；`moe_implementation=fused` 时用 `Qwen2FusedExperts` 做 group-gemm，推理无梯度时还能走 `robby_moe_forward` 融合 kernel。

> **一句话理解**：每个动作 token 经 fp32 gate 打分（sigmoid）→ 加纠偏 bias 选 top-4 专家 → 用原始分数加权求和，再叠加一个所有 token 共享的 shared expert；负载用「bias 纠偏（loss-free）」或「序列级均衡 loss + z-loss」来防止死专家。

---

### 问题 5：双查询蒸馏在代码中如何处理（新手向）

**先建立直觉**：蒸馏（distillation）= 用一个已经训好的大模型（**teacher**）产出的结果，去监督正在训练的小模型（**student**）。「双查询」= 有两组查询 token，分别对齐两个 teacher：**深度**（几何）和**未来视频**（语义时序）。目的是逼模型在生成动作的同时，隐式学会「当前场景的 3D 结构」和「场景接下来会怎么变」——这两个能力对机器人操作至关重要，但直接标注代价极高，于是用现成 teacher 免费监督。

#### (1) 两个 teacher 是谁
- **深度 teacher**：MoGe-2 / MoRGBD，输入当前帧（和未来帧），输出深度图；
- **视频 teacher**：DINO-Video，输入 `[warmup, current, future]` 帧序列，输出语义 patch/cls 特征。

它们在训练循环里以 `torch.no_grad()` 现场计算 target（[`train_lingbotvla.py` 763-791 行](../lingbot-vla-v2/tasks/vla/train_lingbotvla.py#L763)）：
```python
depth_targets, cls_token = get_depth_target(depth_model_type, (moge_model, morgbd_model), pil_images)
future_depth_targets, _  = get_depth_target(..., future_pil_images)      # 未来帧深度
future_video_target_bundle = get_video_target(video_teacher, pil_images, future_pil_images, ...)
```

#### (2) student 侧的「查询 token」从哪来
如问题 2 所述，`init_depth_heads` / `init_video_heads` 注册了若干**可学习参数**作为 query：
- `self.depth_align_embs`（当前深度查询）
- `self.future_depth_align_embs`（未来深度查询）
- `self.future_video_align_embs` / `current_video_align_embs`（视频查询）

它们在 `embed_prefix` 里被拼进 prefix 序列（顺序见 `prefix_query_segments`），在双流注意力中「吸收」图像/语言信息。

#### (3) 用头解码 + 算蒸馏 loss
模型 forward 后，[`FlowMatchingV2.forward` 846-889 行](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L846)：
```python
loss_depth, loss_future_depth, depth_preds, future_depth_preds = \
    self.depth_emb_forward(outputs_embeds, depth_targets, img_masks, future_depth_targets)
loss_depth *= align_params["depth_loss_weight"]              # 0.004
loss_future_depth *= align_params["future_depth_loss_weight"]# 0.004

if self.use_future_video:
    loss_video, future_video_preds, video_metrics = \
        self.video_emb_forward(outputs_embeds, future_video_targets, ...)
    # 可选：当前帧视频对齐 current_video_emb_forward
    loss_future_video = video_total_loss * video_weight       # 0.004
```
- [`depth_emb_forward`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py#L1418)：取出 depth query token → 拼上 image 特征 → 过 `depth_align_head`（`TaskTokenDepthHead`）解码 → 与 teacher 深度算 loss；
- `video_emb_forward`：取出 video query token → 过视频头 → 与 DINO teacher 特征算 loss（支持 patch loss / cosine loss / cls loss）。

#### (4) 关键工程细节
- **loss 权重很小（0.004）**：蒸馏是「辅助任务」，不能盖过主动作 loss；
- **detach 图像特征**：`detach_future_image_feats=True`，未来分支不回传梯度到图像编码，省显存、稳定训练；
- **共享 query 投影**：`share_future_depth_query` / `use_shared_future_task_proj` 让未来深度与未来视频共用查询，减少参数；
- **attention 阻断**：`block_future_depth_to_action` / `block_suffix_to_future_video` 可切断「动作 suffix → 未来 query」的注意力边（用 [`block_suffix_to_fv_`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/utils.py#L172) 在已建好的 2D mask 上把矩形块置 False），保证未来预测不「泄漏」给当前动作生成，维持因果性；
- **推理不需要 teacher**：`align_params` 只在训练用；推理时这些 query token 仍参与前向（因为权重已训练进去），但不算 loss、不需要 MoGe/DINO。

> **一句话理解**：在 prefix 里插两组「空白查询 token」，让它们在注意力中吸收观测信息；训练时用深度头/视频头把它们解码，分别与 MoGe 深度、DINO 视频特征对齐（小权重辅助 loss）。等于用两个现成 teacher 免费教会模型「看懂 3D 几何」和「预判未来变化」，推理时不需要 teacher。

---

### 问题 6：`base_dataset.py`（152-153）如何处理输入数据，输入包含什么

**先回答「输入包含什么」**：π₀ 主要是 image + language（+ state/action）。LingBot-VLA 2.0 的模态明显更丰富。[`VLADataset.getitem`](../lingbot-vla-v2/lingbotvla/data/vla_data/base_dataset.py#L268) 经 `FeatureTransform.apply` 后，[最终 batch_dict](../lingbot-vla-v2/lingbotvla/data/vla_data/utils.py#L468) 包含：

| 字段 | 含义 | π₀ 是否有 |
|------|------|-----------|
| `images` | 多相机图像（如 top/left_wrist/right_wrist）经 Qwen3-VL image_processor 处理 | ✅ |
| `img_masks` | 每路相机是否有效（缺失相机置 False，注意力忽略） | ✅ |
| `image_grid_thw` | Qwen3-VL 的 patch 网格 (t,h,w)，算位置编码用 | Qwen3-VL 特有 |
| `lang_tokens` / `lang_masks` | 任务指令的 token 及 mask | ✅ |
| `state` | 归一化后的 **55 维统一本体状态** | ✅（但维度小） |
| `actions` | 归一化后的 **55 维动作 chunk**（`chunk_size` 步） | ✅ |
| `action_is_pad` | chunk 尾部补位标记 | ✅ |
| `joint_mask` / `state_joint_mask` / `action_joint_mask` | **哪些关节维有效**（55 维里该本体只用其中一部分） | ❌ 特有 |
| `pil_images` / `future_pil_images` | 原始 PIL 图（供 depth/video teacher 现场算 target） | ❌ 特有 |

比 π₀ 多出来的三类，正对应它的三大升级：**joint_mask**（统一 55 维动作空间）、**image_grid_thw**（Qwen3-VL 主干）、**future_pil_images**（双查询蒸馏）。

#### 数据处理链路（`VLADataset`）

**① 底层 LeRobot 数据集**（[`LeRobotDataset`](../lingbot-vla-v2/lingbotvla/data/vla_data/base_dataset.py#L71)）
继承 LeRobot 的 `LeRobotDataset`，兼容 v2/v3 API。`__getitem__` 里：
- 用 `delta_indices` 取一个时间窗的多帧 state/action；
- `_query_videos` 按时间戳从 mp4 解码视频帧（v3 里多 episode 存在共享 mp4，需加 `from_timestamp` 偏移）；
- 对相机帧做 `image_transforms`（Resize）。

**② 时间窗构造**（[`get_delta_timestamps`](../lingbot-vla-v2/lingbotvla/data/vla_data/base_dataset.py#L231) / [`get_video_delta_timestamps`](../lingbot-vla-v2/lingbotvla/data/vla_data/base_dataset.py#L242)）
- 动作：取 `[0, 1, ..., chunk_size-1]` 帧（一整段未来动作）；
- 若 `actions_convert_from_state`（用 state 差分当 action）：取 `chunk_size+1` 帧 state；
- 若 `use_future_image`：视频额外取 `[0, (chunk_size-1)/fps]` 两个时刻（当前帧 + 未来帧），供蒸馏。

**③ 特征变换**（`FeatureTransform.apply`，[`utils.py`](../lingbot-vla-v2/lingbotvla/data/vla_data/utils.py#L375)）
这是「原始 LeRobot 数据 → 模型输入」的核心，四步：
1. `convert_features`：按 `robot_configs/<name>.yaml` 把原始 key（如 `observation.state` 的 0-6 维）映射到统一命名（`observation.state.arm.position` 等）；
2. `action -= state`（若 `subtract_state`）：把绝对动作转成相对当前 state 的增量（末端位姿还支持四元数相对）；
3. `normalizer.normalize`：用 `assets/norm_stats/<name>.json` 归一化；
4. `pad_and_concat` + `prepare_state/prepare_action/prepare_images/prepare_language`：拼接成 55 维、padding、生成 joint_mask、走 Qwen3-VL 图像/文本 processor。

**④ 逆过程** `unapply`（推理输出用）：反归一化 → `action += state` 还原绝对动作 → `reverse_features` 还原成机器人原始 key。

> **一句话理解**：`VLADataset` 在 LeRobot 之上包了一层——底层按时间窗取多帧图像/状态/动作（含未来帧），`FeatureTransform` 依据机器人 yaml 把异构原始数据映射进 55 维统一空间、归一化、生成 joint_mask。输入 = 图像 + 语言 + 55 维 state + 55 维动作 chunk + 三种 mask + 供蒸馏的原始 PIL 图，比 π₀（image+language 为主）多了统一动作空间、Qwen3-VL 网格、未来帧蒸馏三大块。

---

### 问题 7：Muon / FSDP2 优化做了什么，怎么理解

这是两个**正交**的东西：**FSDP2 管「显存怎么分」，Muon 管「参数怎么更新」**。

#### (1) FSDP2 —— 分布式显存分片
FSDP（Fully Sharded Data Parallel）2 是 PyTorch 的分片数据并行。核心思想：把每个 transformer 层的**参数、梯度、优化器状态**切成 N 份分到 N 张卡，需要时才 all-gather 拼回来算，算完立刻释放。这样单卡只存 1/N，能塞下大模型。

[`build_parallelize_model`](../lingbot-vla-v2/lingbotvla/distributed/torch_parallelize.py#L80) 做的事：
- 用 `torch.distributed._composable.fsdp.fully_shard` **逐层包裹**（decoder block 粒度），而不是整模型一刀切；
- `vlm_fsdp=True`：给 Qwen3-VL 主干的 decoder 层和 vision block 单独分片（[`vlm_fsdp` 分支](../lingbot-vla-v2/lingbotvla/distributed/torch_parallelize.py#L340)）；
- **MoE 专家单独分片（EP）**：`Qwen2FusedExperts` 用 `fully_shard(expert_mod, ep_fsdp_kwargs)`，并 `set_gradient_divide_factor`；还建 `_ep_param_set` 供专门的 EP 梯度裁剪（`ep_fsdp2_clip_grad_norm`）；
- 混合精度 `MixedPrecisionPolicy`，可选激活重计算（gradient checkpointing）省显存。

配置里 `data_parallel_mode: fsdp2`、`module_fsdp_enable: true`、`ulysses_parallel_size`（序列并行）都是这套体系。**分布式 Muon**（`robotwin_dist_muon.yaml`）进一步把 Muon 的正交化计算也分摊到各卡并与通信重叠。

#### (2) Muon —— 一种新的优化器（替代 AdamW）
AdamW 对每个参数做逐元素的一阶/二阶矩估计。**Muon 的核心是「对梯度做正交化」**：把梯度矩阵 `G` 通过 **Newton-Schulz 迭代**（[`batched_newton_schulz`](../lingbot-vla-v2/lingbotvla/optim/muon.py#L79)）近似变成正交矩阵 `U`（`U ≈ UVᵀ`，来自 SVD 分解 `G=UΣVᵀ`），再用 `U` 更新参数：

```python
# 五次 Newton-Schulz 迭代（a,b,c 为固定系数）
for _ in range(ns_steps):
    A = ortho @ ortho.mT
    gram_update = torch.addmm(A, A, A, beta=b, alpha=c)
    ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
```
直觉：正交化把各方向的更新幅度拉平（谱归一化），对矩阵型参数（Linear 权重）收敛更好、loss 更低。

**但不是所有参数都用 Muon**——[`split_muon_adamw_params`](../lingbot-vla-v2/lingbotvla/optim/muon.py#L144) 做划分：
- **走 Muon**：2D 权重矩阵、3D MoE 专家堆叠（`gate_proj/up_proj/down_proj`）；
- **走 AdamW**：`nn.Embedding`、`lm_head`、`embed_tokens`、以及各种 1D 参数（norm/bias）——因为这些不适合正交化。

`DistributedMuon`（[muon.py 顶部注释](../lingbot-vla-v2/lingbotvla/optim/muon.py#L1)）针对 FSDP2 分片做了优化：**同形状参数 mega-batch**——堆叠成一个张量、一次 NCCL all-gather、一次批量 NS 迭代、再 scatter 回本地，避免逐参数通信。

#### (3) 两者如何协作
训练一步里（[`train_lingbotvla.py` 880-894 行](../lingbot-vla-v2/tasks/vla/train_lingbotvla.py#L880)）：
```python
grad_norm = ep_fsdp2_clip_grad_norm(model, max_grad_norm)  # FSDP2/EP 感知的梯度裁剪
optimizer.step()      # Muon 对矩阵参数正交化更新 + AdamW 对 embedding 等更新
lr_scheduler.step()
optimizer.zero_grad()
```
MoE 专家还有独立 LR 缩放（`get_moe_param_groups`，scale=√(num_experts/top_k)）。README 提醒：Muon 收敛更好但更慢，注释掉 `optimizer: muon` 即回退 AdamW。

> **一句话理解**：FSDP2 = 把参数/梯度/优化器状态按层分片到多卡省显存（还给 MoE 专家做了专门的 EP 分片）；Muon = 对矩阵型参数的梯度做 Newton-Schulz 正交化再更新（embedding/norm 仍交给 AdamW）。一个管「显存怎么分」，一个管「参数怎么更新」，二者叠加支撑起 6B 模型的分布式高效训练。

---

### 问题 8：FlexAttention 的 mask 构造

FlexAttention 是 PyTorch 2.5+ 的可编程注意力，允许用「mask 函数」表达任意稀疏注意力模式，并编译成高效 kernel。本项目的 mask 构造分**两层**：先造语义 2D mask，再转成 FlexAttention 的 BlockMask。

#### (1) 第一层：`make_att_2d_masks` 造语义 mask（prefix-LM 风格）
[`make_att_2d_masks`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/utils.py#L64) 借自 big_vision，用一个 `mask_ar`（即 `att_masks`）的**前缀和**技巧表达多种注意力：
```python
cumsum = torch.cumsum(att_masks, dim=1)
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]   # 累积 mask_ar 小于等于自己 → 可见
att_2d_masks = att_2d_masks & pad_2d_masks                # 再 AND 上 padding 有效性
```
语义：
- `att_masks=0` 的 token 与前一个共享可见性（**双向**）；
- `att_masks=1` 的 token 开启一个新的因果块（**后面的看不到它之前的块外内容**）。

在本项目里：
- **prefix**（图/文/蒸馏 query）：`vlm_causal=True` 时全 1（因果），否则全 0（双向）；
- **suffix**（state + action）：[`embed_suffix`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py#L1164) 里 `att_masks[:, :2]=True`，即 state token 和第一个 action token 立因果边界，**动作 token 之间双向**、但 **prefix 看不到 suffix**（保证生成动作时不污染观测编码）。

这套 cumsum 方案表达不了「某 query 不能看它前面的某段」，所以有了下面的补丁。

#### (2) 补丁：矩形阻断（蒸馏因果隔离）
[`block_suffix_to_fv_`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/utils.py#L172) 直接在已建好的 2D mask 上把某个矩形块置 False：
```python
att_2d_masks[:, suffix_row_start:, fv_start:fv_end] = False
```
用途：切断「动作 suffix → future-video/future-depth query」这条注意力边，防止未来预测信息泄漏进当前动作生成。支持 `drop_mask` 做逐样本随机阻断（FV-attention dropout），且用广播乘法保持 torch.compile 静态图。

#### (3) 第二层：转成 FlexAttention 的 BlockMask
2D bool mask 不能直接喂 kernel，要转成块稀疏的 `BlockMask`。有两条路径：

**A. `flex_attention_forward`（每层现算，`attention_implementation="flex"`）** —— [`flex_attention.py`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/flex_attention.py#L34)
1. Q/K/V transpose 成 `[B,H,L,D]` 并转 fp32；
2. 序列长度 pad 到 `FLEX_SPARSE_BLOCK_SIZE=128` 的整数倍（FlexAttention 要求）；
3. `precomputed_mask_factory` 把预计算 mask 包成 `mask_mod(b,h,q_idx,kv_idx)` 函数；
4. `create_mask` 生成稠密 4D mask → `create_block_mask` 压成块稀疏 BlockMask；
5. `flex_attention(..., block_mask, enable_gqa=True)` 执行（GQA：32 query 头 / 8 KV 头）。

**B. `build_block_mask` + `flex_attention_with_block_mask`（预建复用，`attention_implementation="flex_cached"`，V2 默认）** —— [`flex_attention.py`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/flex_attention.py#L133)
关键优化：mask 在整个 forward 里**所有 36 层都一样**，没必要每层重建。于是在 [`QwenvlWithExpertV2Model.forward` 363-374 行](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L363)：
```python
if layer_idx == 0:                        # 只在第 0 层建一次
    _full_block_mask = build_block_mask(attention_mask, num_heads, L, L)
att_output = flex_attention_with_block_mask(q, k, v, _full_block_mask, L)  # 后续层复用
```
`build_block_mask` 用 `@torch.compiler.disable` 标注（建 mask 是控制流重的操作，不宜编译），`flex_attention_with_block_mask` 则省掉每次 `create_mask` 的显存分配，直接复用。

#### (4) 推理时的 mask 差异
推理去噪循环里（[`predict_velocity`](../lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L1004)），prefix 已缓存进 KV cache，suffix 只需对 prefix 全可见 + suffix 内部按规则可见，mask 拼成：
```python
prefix_pad_2d_masks = prefix_pad_masks[:,None,:].expand(B, suffix_len, prefix_len)  # suffix 能看全部有效 prefix
full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
```

> **一句话理解**：mask 分两层——先用 `make_att_2d_masks` 的「累积 mask_ar 前缀和」造出 prefix-LM 语义 mask（prefix 双向/因果可配、suffix 动作内双向但看不到 prefix），必要时用 `block_suffix_to_fv_` 矩形阻断蒸馏 query；再通过 `build_block_mask` 转成 FlexAttention 的块稀疏 BlockMask，且 V2 默认只在第 0 层建一次、36 层复用（flex_cached），配合 GQA 高效执行。

---

## 四、快速索引（按问题定位源码）

| 问题 | 关键源码 |
|------|----------|
| 1 policy | `deploy/lingbot_vla_v2_policy.py` L56/L180/L184/L461 |
| 2 depth 处理 | `modeling_lingbot_vla_v2.py` L499/L582；`modeling_lingbot_vla.py` L774/L1418 |
| 3 flow matching | `modeling_lingbot_vla_v2.py` L767/L920/L971；`pi0/modeling_pi0.py` L2082 |
| 4 MoE 路由 | `qwen2_action_expert.py` L222；`moe_load_balance.py`；`modeling_lingbot_vla_v2.py` L170/L1073 |
| 5 双查询蒸馏 | `modeling_lingbot_vla.py` L774/L1418/L1445；`utils.py` L97/L172 |
| 6 数据输入 | `base_dataset.py` L152/L268；`data/vla_data/utils.py` L375/L468 |
| 7 Muon/FSDP2 | `optim/muon.py` L79/L144；`distributed/torch_parallelize.py` L80/L340 |
| 8 FlexAttention mask | `utils.py` L64；`flex_attention.py` L34/L133/L184 |

---

*本文档所有结论均基于对 `lingbot-vla-v2` 源码的直接阅读与核实（含与同仓库 π₀ 实现的逐行对比）。*
