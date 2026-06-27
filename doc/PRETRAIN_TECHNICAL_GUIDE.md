# 预训练模型技术文档

本文档记录 `code/PLM` 目录下预训练语言模型的技术方案、运行流程和后续维护入口。项目仍在完善中，因此本文档采用模块化结构，后续可以按章节增量更新。

## 1. 模块概览

`code/PLM` 当前包含以下模块：

- `config.py`：维护项目路径，如数据目录、tokenizer 目录、日志目录和模型保存目录。
- `train_tokenizer.py`：基于 JSONL 文本训练 BPE tokenizer，并生成 Hugging Face tokenizer 配置文件。
- `dataset.py`：将 JSONL 文本转换为自回归语言模型训练样本，输出 `X`、`Y` 和 `loss_mask`。
- `k_model.py`：实现 Tiny-K causal language model，包含 RMSNorm、RoPE、GQA attention、MLP、DecoderLayer 和 Transformer。
- `ddp_model.py`：预训练入口，负责加载 tokenizer、初始化模型、构建 DataLoader、训练、记录日志和保存 checkpoint。
- `download_dataset.sh`：数据集下载脚本示例。

## 2. 环境依赖

建议先创建独立 Python 环境，再安装依赖：

```bash
pip install -r requirements.txt
```

当前训练入口依赖：

- PyTorch
- Transformers
- Tokenizers
- NumPy
- Pandas
- SwanLab

## 3. 数据格式

预训练数据建议使用 JSONL，每行一个样本：

```json
{"text": "这里是一段用于预训练的文本。"}
```

约定：

- 必须包含 `text` 字段。
- 文本编码使用 UTF-8。
- 空行会被跳过。
- 当前 `PretrainDataset` 会一次性将样本转换为 tensor，适合中小规模数据；大规模语料建议后续改为 mmap、IterableDataset 或离线 tokenized bin 格式。

## 4. Tokenizer

默认 tokenizer 路径为：

```text
code/PLM/tokenizer_k
```

训练/验证脚本：

```bash
python code/PLM/train_tokenizer.py
```

当前脚本默认执行 tokenizer 验证。如果需要重新训练 tokenizer，可在 `train_tokenizer.py` 的 `main()` 中启用 `train_tokenizer(...)` 调用，并确认 `data_path` 指向 JSONL 文件而不是目录。

特殊 token 约定：

- `<unk>`
- `<s>`
- `</s>`
- `<|im_start|>`
- `<|im_end|>`

## 5. 模型结构

`k_model.py` 实现的是 decoder-only causal language model，核心组件包括：

- Token Embedding
- RoPE 旋转位置编码
- Grouped Query Attention
- RMSNorm
- SwiGLU 风格 MLP
- tied embedding / lm head

默认 `ModelConfig` 关键参数：

```python
dim = 768
n_layers = 12
n_heads = 16
n_kv_heads = 8
vocab_size = 6144
max_seq_len = 512
```

训练入口 `ddp_model.py` 当前覆盖为：

```python
dim = 1024
n_layers = 18
```

如果 tokenizer 的 vocab size 变化，需要同步检查 `ModelConfig.vocab_size` 是否一致。

## 6. 预训练流程

准备数据：

```text
code/PLM/datasets/train.json
```

启动训练示例：

```bash
cd code/PLM
python ddp_model.py --data_path datasets/train.json --epochs 1 --batch_size 8 --device cpu --gpus ""
```

GPU 示例：

```bash
cd code/PLM
python ddp_model.py --data_path datasets/train.json --epochs 1 --batch_size 64 --gpus "0,1"
```

常用参数：

- `--epochs`：训练轮数
- `--batch_size`：单步 batch size
- `--learning_rate`：学习率
- `--accumulation_steps`：梯度累积步数
- `--grad_clip`：梯度裁剪阈值
- `--warmup_iters`：warmup 步数
- `--save_interval`：checkpoint 保存间隔
- `--save_path`：checkpoint 保存目录
- `--use_swanlab`：启用 SwanLab 实验跟踪

## 7. Checkpoint

默认模型保存目录来自 `config.py`：

```text
code/PLM/models
```

保存文件命名格式：

```text
pretrain_{dim}_{n_layers}_{vocab_size}.pth
pretrain_{dim}_{n_layers}_{vocab_size}_step{step}.pth
```

## 8. 当前初始化检查结果

已完成：

- `code/PLM/*.py` 通过 `python -m py_compile` 基础语法检查。
- 修复 `ddp_model.py` 中模块路径变量误用问题。
- 修复 `ddp_model.py` 中 `PretrainDataset` 参数名不匹配问题。
- 修复 checkpoint 保存路径使用不存在的 `args.save_dir` 问题。
- 修复 `dataset.py` 中字符串再次 `.decode()` 的问题。
- 为 `pad_token_id` 增加默认兜底。

当前环境限制：

- 当前 `python` 解释器缺少 `torch`，因此 `python code/PLM/ddp_model.py --help` 暂时无法运行。安装 `requirements.txt` 后需要重新验证。

仍建议后续处理：

- 统一 `code/PLM` 文件编码和注释，避免不同终端下中文显示异常。
- 继续统一 `ddp_model.py` 中训练循环相关注释和命名，保持一个清晰版本。
- 增加最小化 smoke test：tokenizer 加载、dataset 单样本、model forward、单步训练。
- 将大语料加载改为流式或离线 tokenized 格式，避免一次性加载占用过多内存。
- 明确 `vocab_size` 与 tokenizer 词表大小的同步策略。

## 9. 后续更新方式

新增能力时建议按以下顺序更新本文档：

1. 在“模块概览”补充新增文件职责。
2. 在“预训练流程”补充新的命令或参数。
3. 在“当前初始化检查结果”移动已完成事项。
4. 在 `doc/CHANGELOG.md` 记录变更。
