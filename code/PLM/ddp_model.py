import argparse
import math
import os
import time
import warnings
from contextlib import nullcontext

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import swanlab
from config import data_dir, model_dir, tokenizer_dir
from dataset import PretrainDataset
from k_model import ModelConfig, Transformer


warnings.filterwarnings("ignore")


def logger(content):
    print(f"日志记录：{content}")


def get_lr(it, total_steps):
    """
    计算当前迭代的学习率，使用余弦退火调度策略
    
    学习率调度策略：
    1. Warmup阶段：学习率从0线性增长到目标学习率
    2. 余弦退火阶段：学习率按余弦函数衰减到最小学习率
    3. 超出训练步数后：保持最小学习率
    
    Args:
        it (int): 当前迭代步数
        total_steps (int): 总迭代步数
        
    Returns:
        float: 当前步数对应的学习率
    """
    warmup_iters = args.warmup_iters
    lr_decay_iters = max(total_steps, warmup_iters + 1)
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / max(1, warmup_iters)

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    decay_ratio = min(max(decay_ratio, 0), 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch):
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            remaining_minutes = spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
            logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    remaining_minutes,
                )
            )

            if args.use_swanlab:
                swanlab.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                    }
                )

        if (step + 1) % args.save_interval == 0:
            save_checkpoint()

        if (step + 1) % 20000 == 0:
            save_checkpoint(step=step + 1)


def save_checkpoint(step=None):
    model.eval()
    suffix = f"_step{step}" if step is not None else ""
    checkpoint_path = os.path.join(
        args.save_path,
        f"pretrain_{llm_config.dim}_{llm_config.n_layers}_{llm_config.vocab_size}{suffix}.pth",
    )
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict, checkpoint_path)
    model.train()


def init_model():
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is not None:
        llm_config.pad_token_id = tokenizer.pad_token_id

    model = Transformer(llm_config)

    if torch.cuda.device_count() > 1:
        logger(f"检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel")
        model = torch.nn.DataParallel(model)

    model = model.to(args.device)
    logger(f"LLM 总参数量：{count_parameters(model) / 1e6:.3f} 百万")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预训练模型参数设置")

    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")

    parser.add_argument("--use_swanlab", action="store_true", help="是否使用 SwanLab 进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载工作进程数")
    parser.add_argument("--data_path", type=str, default=os.path.join(data_dir, "train.json"), help="训练数据路径")

    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="预热迭代次数")

    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--save_path", type=str, default=model_dir, help="模型保存路径")

    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="使用的 GPU ID，用逗号分隔")

    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.use_swanlab:
        swanlab.init(project="PLM", name="pretrain_model", config=vars(args))
        logger("SwanLab 实验跟踪已初始化")

    llm_config = ModelConfig(
        dim=1024,
        n_layers=18,
    )

    os.makedirs(args.save_path, exist_ok=True)
    torch.manual_seed(42)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    model, tokenizer = init_model()
    train_ds = PretrainDataset(args.data_path, tokenizer, max_len=llm_config.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    iter_per_epoch = len(train_loader)

    for epoch in range(args.epochs):
        train_epoch(epoch)
