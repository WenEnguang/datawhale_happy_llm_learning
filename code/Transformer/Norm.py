
import torch
import torch.nn as nn


# 保证数据的分布是均匀分布的
class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Parameter(torch.ones(config['emb_dim']))
        self.l2 = nn.Parameter(torch.zeros(config['emb_dim']))
        self.norm_eps = config['norm_eps']

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.l1 * (x - mean) / (std + self.norm_eps) + self.l2
    
class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Parameter(torch.ones(config['emb_dim']))
        self.norm_eps = config['norm_eps']

    def forward(self, x):
        mean_square = x.pow(2).mean(-1, keepdim=True)
        return self.l1 * x / (mean_square + self.norm_eps).sqrt()