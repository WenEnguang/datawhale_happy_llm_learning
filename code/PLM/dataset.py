import json
import random
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

# 原始方案
class PretrainDataset_1(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 预计每行的起始字节偏移量
        self.offsets = []
        with open(data_path, 'r', encoding='utf-8') as f:
            self.offsets.append(0)
            while f.readline():
                self.offsets.append(f.tell())
        self.total_lines = len(self.offsets) - 1    # 最后一个tell()是EOF
    
    def __len__(self):
        return self.total_lines
    
    def __getitem__(self, idx):
        with open(self.data_path,'r',encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline().strip()
        sample = json.loads(line)
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        input_id =  self.tokenizer(text).data['input_ids'][:self.max_len]
        text_len = len(input_id)

        # 不满足最大长度的剩余部分需要进行填充
        padding_len = self.max_len - text_len
        input_id = input_id + [self.padding] * padding_len
        # padding的位置不需要计算loss
        loss_mask = [1] * text_len + [0] * padding_len
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
    
# torch.tensor方案（自己比较熟悉）
class PretrainDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_len=512):
        super().__init__()
        self.data_path = data_path
        self.max_len = max_len
        bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else ''
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self.all_ids = []   # 存放所有样本的token id list
        self.all_masks = [] # 存放所有样本的loss mask list

        with open(data_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                text = f"{bos_token}{json.loads(line)['text']}"
                ids = tokenizer(
                    text,
                    add_special_tokens=False,
                    max_length=max_len,
                    truncation=True,
                ).input_ids

                text_len = len(ids)
                padding_len = max_len - text_len
                ids = ids + [pad_token_id] * padding_len
                loss_mask = [1] * text_len + [0] * padding_len

                self.all_ids.append(ids)
                self.all_masks.append(loss_mask)
        
        # 一次性转为二维tensor，之后__getitem__直接索引即可
        # shape: (num_samples, max_len)
        self.all_ids = torch.tensor(self.all_ids, dtype=torch.long)
        self.all_masks = torch.tensor(self.all_masks, dtype=torch.long)
    
    def __len__(self):
        return len(self.all_ids)
    def __getitem__(self, idx):
        # 直接索引tensor，速度更快
        ids = self.all_ids[idx]     # shape: (max_len,)
        masks = self.all_masks[idx] # shape: (max_len,)

        X = ids[:-1]     # shape: (max_len-1,)
        Y = ids[1:]      # shape: (max_len-1,)
        loss_mask = masks[1:]  # shape: (max_len-1,)
        return X, Y, loss_mask

# tokenize tensor方案（后续可以考虑使用这个方案，速度更快）
class PretrainDataset_3(Dataset):
    pass

class SFTDataset(Dataset):
    pass
