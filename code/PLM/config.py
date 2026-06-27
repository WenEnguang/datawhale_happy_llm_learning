

import os
from pathlib import Path
import torch

root_dir = Path(__file__).parent
# print(f"项目根目录: {root_dir}")

data_dir = os.path.join(root_dir, "datasets")
# print(f"all数据目录: {data_dir}")

tokenizer_dir = os.path.join(root_dir, "tokenizer_k")
# print(f"分词器目录: {tokenizer_dir}")

log_dir = os.path.join(root_dir, "logs")
# print(f"日志目录: {log_dir}")

model_dir = os.path.join(root_dir, "models")
# print(f"模型目录: {model_dir}")

# 训练参数配置
class Config:
    def __init__(self):
        self.batch_size = 32  # 批量大小
        self.learning_rate = 5e-5  # 学习率
        self.num_epochs = 10  # 训练轮数
        self.warmup_iters = 1000  # 预热迭代次数
        self.max_seq_length = 512  # 最大序列长度
        self.model_name_or_path = "bert-base-uncased"  # 模型名称或路径
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用的设备
        self.learning_rate = 5e-5  # 学习率

    