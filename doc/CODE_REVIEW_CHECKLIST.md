# 代码检查清单

本文档用于在预训练前快速检查项目代码，目标是提前发现会导致训练中断、loss 异常、显存异常或结果不可复现的问题。

## 1. 训练前必查

- Python 文件能否通过语法检查：

```bash
python -m py_compile code/PLM/config.py code/PLM/dataset.py code/PLM/k_model.py code/PLM/ddp_model.py code/PLM/deal_dataset.py code/PLM/train_tokenizer.py
```

- `code/PLM/datasets/train.json` 是否存在。
- 训练数据是否为 JSONL，每行是否包含 `text` 字段。
- `code/PLM/tokenizer_k` 是否包含 `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`。
- tokenizer 词表大小是否与 `ModelConfig.vocab_size` 一致。
- `pad_token_id`、`bos_token`、`eos_token` 是否符合预训练目标。
- `max_seq_len` 是否与数据集、显存预算匹配。
- checkpoint 保存目录是否可写。

## 2. Dataset 检查

- `X`、`Y`、`loss_mask` shape 应一致，长度为 `max_len - 1`。
- padding 位置的 `loss_mask` 应为 0。
- 空行、非法 JSON、缺失 `text` 字段需要有明确处理策略。
- 大规模数据不要一次性全部加载进内存。

## 3. Model 检查

- `dim % n_heads == 0`。
- `n_heads % n_kv_heads == 0`。
- RoPE 缓存长度不小于训练序列长度。
- causal mask 与 padding mask 同时存在时，注意 mask 语义是否正确。
- logits shape 应为 `[batch, seq_len, vocab_size]`。

## 4. Training Loop 检查

- `loss` 是否只在有效 token 上计算。
- 梯度累积时 loss 缩放是否正确。
- 学习率 warmup 为 0 时不能除以 0。
- checkpoint 路径变量必须与 argparse 参数一致。
- CPU 训练时不要强制启用 CUDA AMP。
- 多卡训练建议后续从 `DataParallel` 迁移到 `DistributedDataParallel`。

## 5. 当前已知风险

- `ddp_model.py` 的学习率调度当前只保留一个 `get_lr`；后续修改训练参数时，需要同步检查 warmup、最小学习率和总步数计算。
- `download_dataset.sh` 中包含固定 Linux 路径，公开前建议改为相对路径或参数化。
- `train_tokenizer.py` 默认 `data_path = config.data_dir` 是目录，重新训练 tokenizer 前需要改成具体 JSONL 文件。
- `PretrainDataset` 当前一次性加载所有数据，语料变大后会产生内存压力。
- 项目中存在 `__pycache__`，开源前建议确认 `.gitignore` 已排除。

## 6. 建议新增测试

- `tests/test_tokenizer.py`：加载 tokenizer 并检查特殊 token。
- `tests/test_dataset.py`：构造 2 行 JSONL，验证输出 shape 和 loss mask。
- `tests/test_model_forward.py`：小配置模型 forward，验证 logits 和 loss。
- `tests/test_train_step.py`：CPU 单 batch 反向传播，验证参数可更新。
