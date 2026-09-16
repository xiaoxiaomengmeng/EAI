# π₀ 全流程 torch 函数详解（PyTorch 初学者视角）

> 版本 v1.0 · 2026-09-15 · 源码基线：lerobot `main` v0.6.2（`policies/pi0`）
>
> **配套文档**：[26-PI0模型原理与源码对照详解-v1.0.md](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/docs/26-PI0模型原理与源码对照详解-v1.0.md)（讲"论文理论如何落到代码行"）
> **本文使命**：把 π₀ **全流程**里用到的每一个 torch 函数/运算**一一罗列**，标注**输入 / 输出 / 用途**，面向 **PyTorch 初学者**。
> **涉及源码**：
> - [modeling_pi0.py](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py)（模型主体）
> - [flow_matching.py](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/flow_matching.py)（噪声/时间/euler 积分）
> - [vla_utils.py](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/vla_utils.py)（掩码/padding）

---

## 关于本文（先读这一段）

### 读法约定

- 所有形状示例统一用 **ALoHa 单相机场景**：
  - `B` = batch size（批量）
  - token 总数 **355** = prefix 304（图像 256 + 语言 48）+ suffix 51（state 1 + action 50）
  - 动作/状态维度 padding 到 **32**（`max_action_dim = max_state_dim = 32`）
  - VLM 主干 width = **2048**，Action Expert width = **1024**，`num_heads=8`、`num_kv_heads=1`、`head_dim=256`
- 每个函数给三件事：**输入 → 输出 → 用途**，用途一栏说明它在 π₀ 的哪一步、干什么。
- 函数按**管线阶段**分五大类，最后有一张"全流程串联图"和三条初学者建议。

### 全流程鸟瞰

```
① 造原料     torch.normal(噪声ε) + Beta.sample(时间τ) + nn.Linear(投影头)
② 感知编码   cat 拼 prefix(图像256+语言48) / suffix(state 1 + action 50)
③ 时间融合   create_sinusoidal_pos_embedding → cat[action, time] → F.silu MLP
④ 双流注意力 view拆头 → transpose → cat合流355 → apply_rotary_pos_emb
             → eager_attention_forward(共享softmax+355×355掩码) → reshape合头
⑤ Flow Match 训练: x_t=τ·ε+(1-τ)·A / u_t=ε-A / F.mse_loss(u_t, v_t)
             推理: torch.no_grad + euler_integrate(x_t+dt·v_t, 10步) + clone KV缓存
⑥ 掩码机制   torch.cumsum(块号) → 广播比较(<=) → torch.where(转加性掩码)
⑦ 跨本体     F.pad(补到32维) + ones_like*(-1)(补相机) + .to(bf16)(精度)
⑧ 落地执行   切片[:, -50:] → transpose → 队列逐帧 popleft
```

---

## 一、张量的创建与随机采样（"造原料"）

这一类负责凭空生成噪声、时间步、投影权重、掩码底色。

### 1.1 `nn.Linear(in_features, out_features)`

- **输入**：一个张量 `(…, in_features)`
- **输出**：`(…, out_features)`
- **用途**：π₀ 所有**投影头**都是它——本质是"矩阵乘法 + 偏置"，把机器人原始数据维度对齐到 Action Expert 的隐藏宽度。
- **源码印证**（[modeling_pi0.py L430-435](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L430)）：

```python
self.action_in_proj   = nn.Linear(32, 1024)   # 含噪动作 → Expert 宽度
self.action_out_proj  = nn.Linear(1024, 32)   # Expert 宽度 → 速度场
self.state_proj       = nn.Linear(32, 1024)   # 本体状态 → Expert 宽度
self.action_time_mlp_in  = nn.Linear(2*1024, 1024)  # 融合 [动作⊕时间]
self.action_time_mlp_out = nn.Linear(1024, 1024)
```

### 1.2 `torch.tensor(scalar, dtype, device)`

- **输入**：Python 标量（如 `1.5`、`0.999`、euler 时间 `1.0`）
- **输出**：0 维张量（标量张量）
- **用途**：把标量变成张量，才能参与后续张量运算。
- **源码印证**（[flow_matching.py L37-38 / L104](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/flow_matching.py#L37)）：

```python
alpha_t = torch.tensor(alpha, dtype=torch.float32)   # Beta 形状参数
time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)  # euler 每步的时间
```

### 1.3 `torch.distributions.Beta(α, β).sample((B,))`

- **输入**：形状参数 α=1.5、β=1.0，采样形状 `(B,)`
- **输出**：`(B,)` 落在 (0,1) 区间的随机数
- **用途**：**时间步 τ 的采样**。π₀ 不用均匀采样，而用移位 Beta 分布，让"高噪声、更难预测"的区域多采样。
- **源码印证**（[flow_matching.py L35-58](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/flow_matching.py#L35)）：

```python
def sample_beta(alpha, beta, bsize, device):
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)

def sample_time_beta(bsize, device, *, alpha, beta, scale, offset):
    time_beta = sample_beta(alpha, beta, bsize, device)
    time = time_beta * scale + offset      # 移位缩放：Beta*s + offset
    return time.to(dtype=torch.float32, device=device)
```

> **注**：Beta 采样在 CPU 上做（`_sample_dirichlet` 未实现 MPS），采完再 `.to(device)` 搬到目标设备。

### 1.4 `torch.normal(mean, std, size, dtype, device)`

- **输入**：均值 0.0、标准差 1.0、目标形状 `(B,50,32)`
- **输出**：同形状的标准正态随机张量
- **用途**：**生成纯高斯噪声 ε**，Flow Matching 的积分起点 x₁。
- **源码印证**（[flow_matching.py L43-51](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/flow_matching.py#L43)）：

```python
def sample_noise(shape, device):
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)
```

### 1.5 `torch.ones(...)` / `torch.ones_like(x)`

- **输入**：形状参数，或一个参考张量 `x`
- **输出**：全 1 张量（`ones_like` 与 `x` 同形状/设备）
- **用途**：① 造 `pad_mask`（全 1 = 都是有效 token）；② `torch.ones_like(img) * -1` 给**缺失相机补 -1**（SigLIP 期望输入在 [-1,1]）。
- **源码印证**（[modeling_pi0.py L546 / L1041](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L546)）：

```python
state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)  # 有效 token
img = torch.ones_like(img) * -1     # 缺失相机补 -1
```

### 1.6 `torch.zeros(...)` / `torch.zeros_like(x)`

- **输入**：形状参数，或参考张量 `x`
- **输出**：全 0 张量
- **用途**：① 造 dummy 张量喂给 `rotary_emb`；② `torch.zeros_like(mask)` 把缺失相机的 mask 置 0 → 注意力里被屏蔽。
- **源码印证**（[modeling_pi0.py L99-105 / L1042](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L99)）：

```python
dummy_tensor = torch.zeros(B, N, head_dim, device=..., dtype=...)   # 喂 rotary_emb
mask = torch.zeros_like(mask)   # 缺失相机 mask 置零
```

### 1.7 `create_sinusoidal_pos_embedding(t, dim, min_period, max_period)`

- **输入**：时间步 `(B,)`、目标维度 `dim`、周期范围（`min_period=4e-3`、`max_period=4.0`）
- **输出**：`(B, dim)` 的正弦位置嵌入
- **用途**：**把标量时间步 τ 编码成高维向量**，才能和动作 token 拼接融合。内部由 `torch.arange / exp / cos / sin` 组合实现。
- **源码印证**（[modeling_pi0.py L551-557](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L551)）：

```python
time_emb = create_sinusoidal_pos_embedding(
    timestep, self.action_in_proj.out_features,
    min_period=self.config.min_period, max_period=self.config.max_period,
    device=timestep.device,
)
```

---

## 二、形状变换（"拼装 355 个 token"）

初学者最容易晕的一类——它们**不改数值，只改形状**。

### 2.1 `x[:, None, None]`（None 索引 = `unsqueeze`）

- **输入**：`time` 形状 `(B,)`
- **输出**：`(B, 1, 1)`
- **用途**：**升维以便广播**。L594 把时间步变成 `(B,1,1)`，就能和 `(B,50,32)` 的 noise 逐元素相乘。
- **等价写法**：`time.unsqueeze(1).unsqueeze(2)`。`None` 就是"在这里插入一个长度为 1 的新轴"。

```python
time_expanded = time[:, None, None]   # (B,) → (B,1,1)
```

### 2.2 `state_emb[:, None, :]`

- **输入**：`(B, 1024)`
- **输出**：`(B, 1, 1024)`
- **用途**：把状态从"一个向量"变成"1 个 token"（中间那维 = 序列长度 1），好和 50 个动作 token 在 `dim=1` 上拼接。
- **源码印证**（[modeling_pi0.py L542](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L542)）：`embs.append(state_emb[:, None, :])`

### 2.3 `torch.cat([a, b, ...], dim=)`

- **输入**：若干张量（除 `dim` 外形状必须一致）、拼接轴 `dim`
- **输出**：沿 `dim` 轴变长的张量
- **用途**：π₀ 里多处**合流**：
  - `dim=2`：拼 Q/K/V（VLM 段 + Expert 段 → 355 长度）
  - `dim=1`：拼 `pad_masks` / `att_masks` / `embs`
  - `dim=2`：拼 `[action_emb, time_emb]`（动作 ⊕ 时间）
- **关键提醒**：**cat 拼的是"哪个轴"取决于 dim**，π₀ 里 `cat(dim=2)` 拼的是**序列长度 304+51=355**，不是特征维（这是网上"4608 误解"的根源）。
- **源码印证**（[modeling_pi0.py L96-98 / L567 / L585-586 / L610-611](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L96)）：

```python
query_states = torch.cat(query_states, dim=2)   # 双流 Q 合流成 355
action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # 动作⊕时间
embs = torch.cat(embs, dim=1)                   # state token + action token
pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
```

### 2.4 `.view(B, N, -1, head_dim)`

- **输入**：`(B, N, 2048)`
- **输出**：`(B, N, 8, 256)`
- **用途**：**拆多头**——把一个 2048 维向量看成 8 个头 × 256 维。`-1` 表示"这一维自动算"。`view` 要求内存连续。
- **源码印证**（[modeling_pi0.py L88-91](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L88)）：

```python
hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

### 2.5 `.transpose(dim0, dim1)`

- **输入**：如 `(B, N, 8, 256)`
- **输出**：交换两轴后 `(B, 8, N, 256)`
- **用途**：把"序列 N"和"头 8"两维交换，变成注意力要求的 `(B, heads, seq, head_dim)` 布局；`select_action` 里也用它把 50 步动作转置后入队。
- **源码印证**：L89 `.transpose(1, 2)`；[L1071](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L1071) `actions.transpose(0, 1)`。

### 2.6 `.reshape(B, -1, 1*8*head_dim)`

- **输入**：注意力输出
- **输出**：`(B, N, 2048)`
- **用途**：**合并多头**（view 的逆操作）。L124 的 `1*8*256 = 2048` 正是文档强调的"8 头 × 256"。`reshape` 比 `view` 更宽松，必要时会复制内存。
- **源码印证**（[modeling_pi0.py L124](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L124)）：`att_output.reshape(batch_size, -1, 1 * 8 * head_dim)`

### 2.7 `.expand_as(x)` / `.expand(B, N)`

- **输入**：小张量（含长度 1 的轴）
- **输出**：广播后的大张量（**不复制内存**，只是视图）
- **用途**：把时间嵌入扩展到和 `action_emb` 同形状再拼接；euler 里把标量时间扩展成 `(B,)`。
- **源码印证**（[modeling_pi0.py L566 / L588](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L566)）：

```python
time_emb = time_emb[:, None, :].expand_as(action_emb)
att_masks = att_masks[None, :].expand(bsize, len(att_masks))
```

### 2.8 切片 `x[:, -chunk_size:]`

- **输入**：`(B, 51, 1024)`
- **输出**：`(B, 50, 1024)`
- **用途**：**取最后 50 个 action token**（丢掉前面 1 个 state token），因为只有动作 token 对应速度场。
- **源码印证**（[modeling_pi0.py L633 / L740](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L633)）：`suffix_out = suffix_out[:, -self.config.chunk_size:]`

### 2.9 `att_2d_masks[:, None, :, :]`

- **输入**：`(B, N, N)`
- **输出**：`(B, 1, N, N)`
- **用途**：给布尔掩码加一个"头"维（长度 1 会广播到 8 个头），变成 transformers 要求的 4D 格式。
- **源码印证**（[vla_utils.py L97](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/vla_utils.py#L97)）。

---

## 三、数学运算（Flow Matching 的"心脏"）

这一类真正在做数值计算，对应论文的三个核心公式。

### 3.1 插值路径 `time_expanded * noise + (1 - time_expanded) * actions`

- **输入**：`(B,1,1)` 的 τ、`(B,50,32)` 的 noise、`(B,50,32)` 的 actions
- **输出**：`x_t` `(B,50,32)`
- **用途**：**A_t^τ = τ·ε + (1−τ)·A_t**。靠广播一次算完整个动作块。约定 τ=1 纯噪声、τ=0 纯动作。
- **源码印证**（[modeling_pi0.py L594-595](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L594)）。

### 3.2 目标速度场 `noise - actions`

- **输入**：两个 `(B,50,32)`
- **输出**：`u_t` `(B,50,32)`
- **用途**：**u = ε − A_t**，直线插值的方向（与 τ 无关），是训练的回归标签。
- **源码印证**（[modeling_pi0.py L596](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L596)）。

### 3.3 euler 积分 `x_t = x_t + dt * v_t`

- **输入**：当前 `x_t`、速度场 `v_t`、步长 `dt = -1/num_steps`（默认 -1/10）
- **输出**：更新后的 `x_t`
- **用途**：**推理时从纯噪声沿速度场积分 10 步回到干净动作**（从 τ=1 走到 τ=0）。
- **源码印证**（[flow_matching.py L100-136](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/flow_matching.py#L100)）：

```python
dt = -1.0 / num_steps
x_t = noise
for step in range(num_steps):
    time = 1.0 + step * dt
    v_t = denoise_fn(x_t, time_tensor)   # 预测速度场
    x_t = x_t + dt * v_t                 # 前向 euler 一步
return x_t
```

### 3.4 `torch.cumsum(x, dim=)`

- **输入**：`(B,N)` 的 0/1 序列或 mask
- **输出**：同形状的**累积和**
- **用途**：两处关键——
  - `cumsum(pad_masks) - 1` 生成**位置编码**（padding 位置不递增）
  - `cumsum(att_masks)` 给每个 token 编**块号**（0=图像语言、1=state、2=action）
- **源码印证**（[modeling_pi0.py L614](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L614)、[vla_utils.py L86](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/vla_utils.py#L86)）：

```python
position_ids = torch.cumsum(pad_masks, dim=1) - 1
cumsum = torch.cumsum(att_masks, dim=1)
```

### 3.5 广播比较 `cumsum[:, None, :] <= cumsum[:, :, None]`

- **输入**：`(B,1,N)` 与 `(B,N,1)`
- **输出**：布尔 `(B,N,N)`
- **用途**：**生成 2D 注意力掩码**。规则"token i 能看 j ⟺ 块号[j] ≤ 块号[i]"，实现"块状因果 + 块内双向"。
- **源码印证**（[vla_utils.py L87](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/vla_utils.py#L87)）。

### 3.6 广播乘（外积）`pad_masks[:, None, :] * pad_masks[:, :, None]`

- **输入**：`(B,1,N)` × `(B,N,1)`
- **输出**：`(B,N,N)`
- **用途**：只有行列**都是有效 token** 时才为 1，把 padding 屏蔽叠加到注意力掩码上。
- **源码印证**（[vla_utils.py L88](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/vla_utils.py#L88)）。

### 3.7 `torch.where(cond, a, b)`

- **输入**：布尔张量 `cond`、两个值/张量 `a`、`b`
- **输出**：逐元素选择的结果（`cond` 为真取 `a`，否则取 `b`）
- **用途**：① 把布尔掩码转成**加性掩码**（可见填 `0.0`，遮蔽填大负数 → softmax 后≈0）；② euler 里 clamp 已知动作前缀（RTC）。
- **源码印证**（[vla_utils.py L98](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/vla_utils.py#L98)、[flow_matching.py L109/L131](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/flow_matching.py#L109)）：

```python
result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
```

### 3.8 `F.mse_loss(u_t, v_t, reduction="none")`

- **输入**：两个 `(B,50,32)`（目标速度 `u_t`、预测速度 `v_t`）
- **输出**：逐元素平方差（`reduction="none"` 不求和/平均，保留形状）
- **用途**：**训练损失 = ‖v_θ − u‖²**。注意是 MSE 回归而非交叉熵——这正是"连续动作、非自回归"的体现。补的零（padding 维度）后续会截断，不参与 loss。
- **源码印证**（[modeling_pi0.py L641](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L641)）。

### 3.9 `F.silu(x)`

- **输入**：`(…, 1024)`
- **输出**：同形状
- **用途**：MLP 里的**激活函数**（`x · sigmoid(x)`），给 `action_time_mlp` 加非线性。
- **源码印证**（[modeling_pi0.py L571](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L571)）。

### 3.10 `F.pad(vector, (0, n))`

- **输入**：`(…, 14)` 的向量、填充量 `(0, n)`（末尾补 n 个 0）
- **输出**：`(…, 32)`
- **用途**：**32 维 padding**——ALOHA 的 14D 动作末尾补 18 个 0，把所有本体塞进同一形状 `(B,50,32)`，这是跨本体预训练的结构基础。
- **源码印证**（[vla_utils.py L116-138](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/common/vla_utils.py#L116)）：

```python
if vector.shape[-1] >= new_dim:
    return vector                                  # 已够长，不动
return F.pad(vector, (0, new_dim - vector.shape[-1]))  # 末尾补零到 32
```

---

## 四、类型 / 设备 / 梯度控制（"工程细节"）

### 4.1 `.to(...)`

- **输入**：任意张量 + 目标 dtype/device（如 `.to(torch.float32)`、`.to(dtype=torch.bfloat16)`、`.to(device)`）
- **输出**：换类型/设备后的张量
- **用途**：**混合精度对齐**——投影前把 bf16 转回 float32 保精度，进 MLP 前再转 bf16 省显存；CPU 采样的 Beta 结果搬到 GPU。
- **源码印证**（[modeling_pi0.py L636 / L140](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L636)）：`v_t = self.action_out_proj(suffix_out.to(torch.float32))`

### 4.2 `.type(dtype=...)`

- **输入**：张量 + 目标 dtype
- **输出**：换 dtype 的张量
- **用途**：把时间嵌入对齐到 timestep 的精度（与 `.to` 类似，语义略不同）。
- **源码印证**（[modeling_pi0.py L558](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L558)）：`time_emb = time_emb.type(dtype=timestep.dtype)`

### 4.3 `.clone()`

- **输入**：张量
- **输出**：独立副本（新内存）
- **用途**：① 残差前保存 `after_first_residual`；② `clone_past_key_values` **复制 KV 缓存**，避免 10 步去噪互相污染。
- **源码印证**（[modeling_pi0.py L136 / L730](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L136)）。

### 4.4 `@torch.no_grad()`

- **输入**：装饰一个函数
- **输出**：函数内所有运算**不追踪梯度**
- **用途**：**推理专用**（`sample_actions`）——不建计算图，省显存、加速。
- **源码印证**（[modeling_pi0.py L643 上方](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L643)）。

### 4.5 `model.eval()`

- **输入**：模型对象
- **输出**：切换到评估模式
- **用途**：`select_action` 开头调用，关掉 dropout / batchnorm 更新等训练态行为。
- **源码印证**（[modeling_pi0.py L1060](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L1060)）：`self.eval()`

---

## 五、注意力算子（HF transformers 侧，双流交汇处）

### 5.1 `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)`

- **输入**：Q、K `(B, heads, 355, 256)` + 旋转系数 `cos/sin`
- **输出**：注入位置信息后的 Q、K（形状不变）
- **用途**：**旋转位置编码 RoPE**——让注意力感知 token 的相对位置。π₀ 连位置编码都借用 VLM 的 `rotary_emb`。
- **源码印证**（[modeling_pi0.py L106-109](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L106)）：

```python
cos, sin = rotary_emb(dummy_tensor, position_ids)
query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
    query_states, key_states, cos, sin, unsqueeze_dim=1)
```

### 5.2 `eager_attention_forward(attn, q, k, v, mask, scaling)`

- **输入**：Q `(B,8,355,256)`、K/V `(B,1,355,256)`（GQA）、4D 加性掩码、缩放系数
- **输出**：注意力输出 `(B, 355, 8, 256)`
- **用途**：**一次共享注意力**——内部做 `QKᵀ/√256 + mask → softmax → ×V`。VLM 和 Expert 的 token 在这一步"同场开会"，靠 355×355 掩码控制谁能看谁。GQA：8 个 query 头共享 1 组 K/V。
- **源码印证**（[modeling_pi0.py L114-121](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/repos/lerobot-main/src/lerobot/policies/pi0/modeling_pi0.py#L114)）。

---

## 六、速查总表（一页记住全部）

| 阶段 | 函数 | 输入 → 输出 | 一句话用途 |
|---|---|---|---|
| 造原料 | `nn.Linear` | `(…,in)`→`(…,out)` | 投影头，对齐维度 |
| 造原料 | `torch.tensor` | 标量→0维张量 | 标量变张量 |
| 造原料 | `Beta.sample` | `(B,)` | 采样时间步 τ |
| 造原料 | `torch.normal` | `size`→同形状 | 生成纯噪声 ε |
| 造原料 | `torch.ones/ones_like` | 形状/参考→全1 | pad_mask、补 -1 相机 |
| 造原料 | `torch.zeros/zeros_like` | 形状/参考→全0 | dummy、mask 置零 |
| 造原料 | `create_sinusoidal_pos_embedding` | `(B,)`→`(B,dim)` | 时间正弦嵌入 |
| 形状 | `x[:,None,None]` | `(B,)`→`(B,1,1)` | 升维广播 |
| 形状 | `torch.cat` | 多张量→拼接 | 合流（序列/特征） |
| 形状 | `.view` | `(B,N,2048)`→`(B,N,8,256)` | 拆多头 |
| 形状 | `.transpose` | 换两轴 | 变注意力布局 |
| 形状 | `.reshape` | →`(B,N,2048)` | 合并多头 |
| 形状 | `.expand_as/.expand` | 小→大（不复制） | 广播扩展 |
| 形状 | 切片 `[:,-50:]` | `(B,51,·)`→`(B,50,·)` | 取动作 token |
| 数学 | `τ*noise+(1-τ)*actions` | →`x_t` | 插值路径 A_t^τ |
| 数学 | `noise-actions` | →`u_t` | 目标速度场 u |
| 数学 | `x_t+dt*v_t` | →`x_t` | euler 积分去噪 |
| 数学 | `torch.cumsum` | `(B,N)`→累积和 | 位置编码 / 块号 |
| 数学 | 广播比较 `<=` | →`(B,N,N)` 布尔 | 2D 注意力掩码 |
| 数学 | 广播乘 `*` | →`(B,N,N)` | padding 外积屏蔽 |
| 数学 | `torch.where` | 布尔+两值→选择 | 加性掩码 / clamp |
| 数学 | `F.mse_loss` | 两张量→平方差 | 训练损失 ‖v-u‖² |
| 数学 | `F.silu` | `(…,1024)`→同形状 | 激活函数 |
| 数学 | `F.pad` | `(…,14)`→`(…,32)` | 32 维补零 |
| 类型 | `.to` | →换 dtype/device | 混合精度对齐 |
| 类型 | `.type` | →换 dtype | 时间嵌入对齐 |
| 类型 | `.clone` | →独立副本 | 残差保存 / KV 复制 |
| 梯度 | `@torch.no_grad` | 关梯度 | 推理省显存 |
| 梯度 | `model.eval` | 评估模式 | 关训练态行为 |
| 注意力 | `apply_rotary_pos_emb` | Q,K→注入位置 | RoPE |
| 注意力 | `eager_attention_forward` | Q,K,V,mask→输出 | 共享 softmax 注意力 |

---

## 七、给初学者的 3 条重点建议

1. **先吃透"广播 (broadcasting)"**：π₀ 里 `time[:, None, None] * noise`、`cumsum[:, None, :] <= cumsum[:, :, None]`、`pad_masks[:, None, :] * pad_masks[:, :, None]` 全是广播。理解"长度为 1 的轴会自动复制对齐"，这些代码就不神秘了。
2. **区分 `view/reshape` 和 `transpose`**：前者改形状、后者换轴顺序。注意力里"拆头（view）→ 换轴（transpose）→ 算（attention）→ 合头（reshape）"这套组合拳是所有 Transformer 的通用套路。
3. **`torch.cat` 的 dim 决定拼什么**：π₀ 里 `cat(dim=2)` 拼的是**序列长度 304+51=355**，不是特征维——这也是网上"18 头 / 4608 共享空间"误解的根源。记住：**cat 前先看 dim**。

---

> **与相邻文档的分工**：本文（27 号）是 π₀ 全流程 **torch 函数字典**（输入/输出/用途，面向 PyTorch 初学者）；[26 号](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/docs/26-PI0模型原理与源码对照详解-v1.0.md)讲**原理 ↔ 源码逐行对照**；[19 号](file:///Users/damon/Documents/10_code/openSources/ws/11_pi_serial_models/docs/19-PI0训练与评估技术数据流详解-v1.0.md)讲**训练/评估数据流**。建议先读 26 号建立整体认知，再用本文当"函数速查手册"。
