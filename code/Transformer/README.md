# Transformer 从零实现 —— 教学级开源项目

一个**纯 PyTorch 从零实现**的 Transformer 模型（"Attention Is All You Need" 架构），包含 Encoder-Decoder 完整结构。每个模块独立成文件，代码注释详尽（中文），适合学习 Transformer 内部机制的初学者。

## 项目结构

```
code/Transformer/
├── config.py        # 模型超参数配置（统一字典 contract）
├── Attention.py     # 多头注意力（MHA）：自注意力 + 因果掩码 + 交叉注意力
├── FFN.py           # 前馈网络（Feed-Forward Network）
├── Norm.py          # 层归一化：LayerNorm + RMSNorm
├── PE.py            # 位置编码：正弦/余弦编码 + RoPE（旋转位置编码）
├── Encoder.py       # 编码器：EncoderLayer + Encoder（N 层堆叠）
├── Decoder.py       # 解码器：DecoderLayer + Decoder（含交叉注意力）
├── Transformer.py   # 顶层模型组装 + 推理示例
└── CLAUDE.md        # 项目开发说明
```

## 架构总览

```
输入 Token IDs  [batch, src_len]
      │
      ├── Embedding + PositionalEncoding + Dropout ──┐
      │                                                │
      ▼                                                ▼
  ┌──────────────────────────────────────────────────────┐
  │                   Encoder (× N)                       │
  │  ┌─────────────────────────────────────────────────┐ │
  │  │  EncoderLayer                                    │ │
  │  │    x = x + MHA( LayerNorm(x) )     ← 自注意力    │ │
  │  │    x = x + FFN( LayerNorm(x) )     ← 前馈网络    │ │
  │  └─────────────────────────────────────────────────┘ │
  │  ... 重复 N 层 ...                                    │
  │  → final LayerNorm → enc_output  [batch, src_len, d] │
  └──────────────────────┬───────────────────────────────┘
                         │
  ┌──────────────────────┼───────────────────────────────┐
  │                   Decoder (× N)                       │
  │  ┌─────────────────────────────────────────────────┐ │
  │  │  DecoderLayer                                    │ │
  │  │    x = x + MHA( LN(x), causal=True )  ← 因果自注意力│
  │  │    x = x + MHA( LN(x), enc_output )  ← 交叉注意力  │ │
  │  │    x = x + FFN( LN(x) )               ← 前馈网络  │ │
  │  └─────────────────────────────────────────────────┘ │
  │  ... 重复 N 层 ...                                    │
  │  → final LayerNorm → dec_output  [batch, tgt_len, d] │
  └──────────────────────┬───────────────────────────────┘
                         │
                    Linear(d, vocab_size)
                         │
                    logits  [batch, tgt_len, vocab_size]
```

---

## 模块详解

### 1. `config.py` —— 超参数配置

整个项目的**唯一配置来源**，所有模块通过同一个字典获取参数。

```python
model_config = {
    'vocab_size': 151936,      # 词表大小（适配 Qwen2.5 tokenizer）
    'context_length': 128,     # 最大序列长度
    'emb_dim': 256,            # 词嵌入维度（残差流宽度）
    'hidden_dim': 512,         # FFN 隐藏层维度（通常为 emb_dim 的 2~4 倍）
    'drop_rate': 0.1,          # Dropout 比例
    'n_layers': 6,             # Encoder / Decoder 层数
    'n_heads': 4,              # 注意力头数（必须能整除 emb_dim）
    'qkv_bias': True,          # Q/K/V 投影是否使用偏置
    'norm_eps': 1e-5,          # LayerNorm 的 epsilon
    'is_causal': True          # 默认是否使用因果掩码
}
```

**设计要点：** 每个模块的 `__init__(self, config)` 都从这个字典读取所需键值。添加新模块时遵循此约定即可无缝接入。

---

### 2. `Attention.py` —— 多头注意力（MHA）

<details>
<summary><b>核心公式</b></summary>

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V$$

- $Q, K, V$：Query / Key / Value 矩阵，形状均为 `[batch, n_heads, seq_len, head_dim]`
- $\sqrt{d_k}$：缩放因子，防止点积方差随维度增大
- `mask`：因果掩码（下三角 `-inf`），阻止未来 token 的信息泄露
</details>

**三种模式：**

| 模式 | `enc_output` | `is_causal` | Q 来源 | K/V 来源 | 用途 |
|------|:-----------:|:-----------:|--------|----------|------|
| 编码器自注意力 | `None` | `False` | x | x | Encoder |
| 解码器因果自注意力 | `None` | `True` | x | x | Decoder 第一层 |
| 交叉注意力 | 传入 | `False` | x | enc_output | Decoder 第二层 |

**因果掩码实现：**

```python
# 构造布尔上三角掩码（j > i 为 True，即"需要屏蔽的未来位置"）
mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
# 将未来位置的分数设为 -inf，softmax 后权重 → 0
attn_scores = attn_scores.masked_fill(mask[:l, :l], float('-inf'))
```

- 掩码注册为 `persistent=False` 的 buffer（不参与 `state_dict`，避免切换 `context_length` 时形状冲突）
- 交叉注意力时**不应用**因果掩码（解码器需要看到全部编码器输出）

---

### 3. `FFN.py` —— 前馈网络

<details>
<summary><b>结构</b></summary>

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

- `fc1`: `emb_dim → hidden_dim`（扩张，如 256 → 512）
- `fc2`: `hidden_dim → emb_dim`（投影回残差流宽度）
- 中间夹 Dropout
</details>

使用 **ReLU** 激活函数（经典 Transformer 风格，非 GPT 系列的 GELU/SwiGLU）。如需切换激活函数，将 `F.relu` 替换为 `F.gelu` 即可。

---

### 4. `Norm.py` —— 层归一化

提供两种实现：

#### LayerNorm

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

- $\mu = \frac{1}{d}\sum x_i$，$\sigma = \sqrt{\frac{1}{d}\sum(x_i - \mu)^2}$（**有偏**方差，除以 N）
- `self.l1` = $\gamma$（缩放），`self.l2` = $\beta$（偏置），均为可学习参数
- 沿最后一维（`emb_dim`）归一化

#### RMSNorm

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}}$$

- 无偏置项，无均值中心化
- LLaMA 系列模型使用，计算更高效

---

### 5. `PE.py` —— 位置编码

#### 正弦/余弦位置编码（`PositionalEncoding`）

$$PE_{(pos, 2i)} = \sin\left(pos / 10000^{2i/d}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i/d}\right)$$

- 固定编码（非可学习），与 Embedding 结果**相加**
- 注册为 `persistent=True` 的 buffer（随模型保存/加载）

#### RoPE（旋转位置编码）—— 进阶内容

文件中同时提供了完整的 **RoPE（Rotary Position Embedding）** 实现，包含：

| 函数/类 | 作用 |
|---------|------|
| `get_rotary_frequency(head_dim, seq_len, theta)` | 生成旋转频率矩阵 $\Theta$ |
| `get_rotary_embedding(head_dim, seq_len, theta)` | 预计算 `cos(m·θ)` / `sin(m·θ)` |
| `rotate_half(x)` | 半向量旋转辅助函数：`[x1,x2,x3,x4] → [-x3,-x4,x1,x2]` |
| `apply_rotary_pos_emb(q, k, cos, sin)` | 对 Q/K 应用 RoPE：$q' = q\cos\theta + \text{rotate\_half}(q)\sin\theta$ |
| `RotaryPositionalEmbedding` | 完整的 RoPE `nn.Module` 封装 |

> **注意：** RoPE 模块当前未接入主模型 `Transformer`。如需使用，将 `PositionalEncoding` 替换为对 Q/K 插入 `RotaryPositionalEmbedding` 即可。

---

### 6. `Encoder.py` —— 编码器

**EncoderLayer（单层）：**

```
x ──→ LayerNorm ──→ MHA（自注意力）──→ + ──→ LayerNorm ──→ FFN ──→ + ──→ 输出
      ↑_________________________________↑    ↑___________________________↑
              残差连接                                    残差连接
```

采用 **Pre-Norm** 结构（归一化在子层之前），这是现代 Transformer（GPT-2、LLaMA）的标准做法。

**Encoder（堆叠）：**
- `n_layers` 个 `EncoderLayer`，通过 `nn.ModuleList` 顺序执行
- 最后接一层 `final_norm`

---

### 7. `Decoder.py` —— 解码器

**DecoderLayer（单层）：**

```
x ──→ LN ──→ Masked MHA（因果自注意力）──→ + ──→ LN ──→ Cross MHA（交叉注意力）──→ + ──→ LN ──→ FFN ──→ + ──→ 输出
       ↑____________________________________↑       ↑______________________________↑       ↑___________________↑
```

三种注意力角色的对比：

| 子层 | 注意力类型 | Q 来源 | K/V 来源 | 掩码 |
|------|-----------|--------|----------|:----:|
| `masked_mha` | 因果自注意力 | decoder 输入 | decoder 输入 | ✅ 因果 |
| `cross_mha` | 交叉注意力 | decoder 输入 | encoder 输出 | ❌ 无 |
| `ffn` | — | — | — | — |

---

### 8. `Transformer.py` —— 顶层模型

**完整数据流：**

```
训练阶段 (target_ids 不为 None)：
  input_ids ──→ Embed ──→ PE ──→ Dropout ──→ Encoder ──→ enc_output
  target_ids ──→ Embed ──→ PE ──→ Dropout ──→ Decoder(·, enc_output) ──→ Linear ──→ logits ──→ CrossEntropyLoss

推理阶段 (target_ids = None)：
  input_ids ──→ ... 同上 ... ──→ enc_output
  input_ids[:, -1:] ──→ Embed ──→ PE ──→ Dropout ──→ Decoder(·, enc_output) ──→ Linear ──→ logits
```

**权重初始化：**
- `nn.Linear` / `nn.Embedding`：正态分布 N(0, 0.02)
- 偏置初始化为 0
- `nn.LayerNorm`：weight=1, bias=0

**参数共享：** 编码器和解码器**共享同一个 Embedding 矩阵**（`self.embed`），这也是原始 Transformer 的做法。

---

## 快速开始

### 环境要求

```bash
pip install torch transformers
```

### 修改 tokenizer 路径

编辑 `Transformer.py` 的 `main()` 函数，将 tokenizer 路径指向你本机的模型目录：

```python
tokenizer = AutoTokenizer.from_pretrained(r"你的/模型/路径")
```

或者直接使用 HuggingFace 在线模型（需要网络）：

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
```

同时需要将 `config.py` 中的 `vocab_size` 改为对应 tokenizer 的词表大小。

### 运行

```bash
cd code/Transformer
python Transformer.py
```

输出示例：
```
Starting main function
number of parameters: 85.85M
Logits shape: torch.Size([1, 151936])
Loss: None
Output: <预测的文本>
```

> **注意：** 当前为未训练模型，输出为随机结果。训练循环（数据集加载、优化器、反向传播、梯度裁剪等）留待后续实现。

---

## 配置参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocab_size` | 151936 | 词表大小，需匹配 tokenizer |
| `context_length` | 128 | 最大序列长度 |
| `emb_dim` | 256 | 残差流维度 |
| `hidden_dim` | 512 | FFN 隐藏层维度 |
| `drop_rate` | 0.1 | Dropout 概率 |
| `n_layers` | 6 | Encoder/Decoder 层数 |
| `n_heads` | 4 | 注意力头数 |
| `qkv_bias` | True | Q/K/V 投影是否带偏置 |
| `norm_eps` | 1e-5 | 归一化 epsilon |
| `is_causal` | True | 默认注意力掩码模式 |

## 设计约定

- **Config dict 契约：** 所有模块构造函数接受 `config` 字典，从中读取所需超参数
- **Pre-Norm 架构：** LayerNorm 在 MHA/FFN 之前（而非之后），梯度流动更稳定
- **张量布局：** `[batch_size, seq_len, hidden_dim]`；注意力内部 reshape 为 `[batch, n_heads, seq_len, head_dim]`
- **中文注释：** 代码注释统一使用中文，降低国内学习者的阅读门槛

## 待实现

- [ ] 训练循环（Teacher Forcing + 交叉熵损失 + 学习率调度）
- [ ] 推理时的自回归解码（KV Cache）
- [ ] RoPE 接入主模型
- [ ] 混合精度训练（AMP）
- [ ] 数据集示例（如 WMT 翻译、WikiText 语言建模）
- [ ] 单元测试

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer 论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm 论文

## License

MIT
