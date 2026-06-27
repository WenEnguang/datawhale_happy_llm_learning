# Datawhale NLP / PLM 教学项目

本项目面向 NLP、Transformer 与预训练语言模型的学习和实践，当前包含两类内容：

- 理论文档：`1、NLP概念.md`、`2、Transformer架构.md`、`3、预训练语言模型.md`、`4、大语言模型训练.md`
- 代码实现：`code/Transformer` 从零实现 Transformer，`code/PLM` 实现小型预训练语言模型相关流程

## 当前状态

已经开源的部分主要是基础理论文档和 Transformer 实现；预训练模型部分仍在整理中，代码位于 `code/PLM`，包含 tokenizer、dataset、模型结构和训练入口。

## 推荐阅读顺序

1. `1、NLP概念.md`
2. `2、Transformer架构.md`
3. `code/Transformer/README.md`
4. `3、预训练语言模型.md`
5. `doc/PRETRAIN_TECHNICAL_GUIDE.md`
6. `code/PLM`

## 预训练模块

预训练流程文档见 `doc/PRETRAIN_TECHNICAL_GUIDE.md`。

环境依赖：

```bash
pip install -r requirements.txt
```

核心文件：

- `code/PLM/config.py`：路径和基础训练配置
- `code/PLM/train_tokenizer.py`：训练与验证 tokenizer
- `code/PLM/dataset.py`：预训练数据集封装
- `code/PLM/k_model.py`：Tiny-K causal language model
- `code/PLM/ddp_model.py`：预训练入口

## 维护约定

为了方便后续持续完善，建议新增功能时同步更新：

- `doc/PRETRAIN_TECHNICAL_GUIDE.md`：预训练流程、命令、参数和数据格式
- `doc/CODE_REVIEW_CHECKLIST.md`：训练前检查项和已知风险
- `doc/CHANGELOG.md`：重要变更记录
