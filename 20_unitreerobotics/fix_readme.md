# RTX 5090 (sm_120) 兼容性修复记录

> 环境：RTX 5090 (compute capability 12.0) + PyTorch 2.11.0+cu128 + Python 3.10.12
>
> 官方代码基于 torch 2.3.1 + xformers 0.0.27 + A100/H100 (sm_80/sm_90) 开发，
> 在 RTX 5090 上需要以下适配修改。

---

## 修复 1：Python 版本约束放宽

**文件**：`pyproject.toml`

**问题**：`requires-python = "==3.10.18"`，而系统 Python 为 3.10.12，pip 拒绝安装。

**修复**：

```toml
# 原
requires-python = "==3.10.18"
# 改为
requires-python = ">=3.10"
```

---

## 修复 2：xformers fa2 不支持 float32 (sm_120)

**文件**：`src/unifolm_wma/modules/attention.py` — `efficient_forward` 方法

**问题**：xformers 的 flash attention 2 后端在 sm_120 上仅支持 fp16/bf16，
当输入为 float32 时抛出：

```
NotImplementedError: No operator found
`memory_efficient_attention` is not supported on the current GPU (float32, sm_120)
```

**修复**：在 `efficient_forward` 入口处检测 dtype，自动 cast 到 bfloat16：

```python
# RTX 5090 (sm_120): xformers fa2 only supports fp16/bf16
_attn_dtype = q.dtype
_use_cast = _attn_dtype not in (torch.float16, torch.bfloat16)
if _use_cast:
    q = q.to(torch.bfloat16)

# 每个 xformers 调用前 cast k/v，调用后 cast 回原 dtype
if _use_cast:
    k, v = k.to(torch.bfloat16), v.to(torch.bfloat16)
out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
out = out.to(_attn_dtype)
```

---

## 修复 3：xformers fa2 不支持 tensor attn_bias (sm_120)

**文件**：`src/unifolm_wma/modules/attention.py` — `efficient_forward` 方法中 `out_aa` 部分

**问题**：agent-action cross attention 使用 tensor 类型的 `attn_bias`（mask），
xformers fa2 在 sm_120 上不支持：

```
NotImplementedError: No operator found
`memory_efficient_attention` is not supported (attn_bias type: Tensor, sm_120)
```

**修复**：将该处 xformers 调用替换为 PyTorch 原生 `scaled_dot_product_attention`：

```python
# Use PyTorch native SDPA for masked attention
_q_4d = q.reshape(b, self.heads, q.shape[1], self.dim_head)
_k_4d = k_aa.reshape(b, self.heads, k_aa.shape[1], self.dim_head).to(q.dtype)
_v_4d = v_aa.reshape(b, self.heads, v_aa.shape[1], self.dim_head).to(q.dtype)
_m_4d = attn_mask_aa.reshape(b, self.heads, attn_mask_aa.shape[1], attn_mask_aa.shape[2])
out_aa = torch.nn.functional.scaled_dot_product_attention(
    _q_4d, _k_4d, _v_4d, attn_mask=_m_4d)
out_aa = out_aa.permute(0, 2, 1, 3).reshape(b, q.shape[1], self.heads * self.dim_head)
out_aa = out_aa.to(_attn_dtype)
```

> **注意**：`k_aa`/`v_aa` 必须 `.to(q.dtype)` 确保与已 cast 为 bfloat16 的 `q` dtype 一致。

---

## 修复 4：torchvision.io.write_video 已移除

**文件**：`scripts/evaluation/world_model_interaction.py` — `save_results` 函数

**问题**：新版 torchvision（≥0.20）移除了 `torchvision.io.write_video`：

```
AttributeError: module 'torchvision.io' has no attribute 'write_video'
```

**修复**：改用已导入的 `imageio`：

```python
# 原
torchvision.io.write_video(filename, grid, fps=fps, video_codec='h264', options={'crf': '10'})
# 改为
imageio.mimwrite(filename, grid.numpy(), fps=fps, quality=9)
```

---

## 修复 5：清除 Python 缓存

修改 `.py` 文件后，如果存在旧的 `__pycache__`，Python 可能仍加载旧字节码：

```bash
find /home/creator/ws/20_unitreerobotics/repos/unifolm-world-model-action/src \
    -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

---

## 依赖安装注意事项

| 问题 | 原因 | 解决 |
|------|------|------|
| `pip install -e .` 覆盖 torch | pyproject.toml pin 了 torch==2.3.1 | 使用 `--no-deps` 安装主包 |
| tensorflow 依赖回溯 | tf-graphics 依赖 tf>=2.2 与 numpy 1.x 冲突 | pin `tensorflow==2.13.1`，tf-graphics 用 `--no-deps` |
| `typing-extensions` 被降级 | tensorflow 安装时拉低版本 | 事后 `pip install "typing-extensions>=4.16.0"` |
| 缺少 pyarrow / datasets | 数据加载需要 | 单独 `pip install pyarrow datasets==3.6.0` |
