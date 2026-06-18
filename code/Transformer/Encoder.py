
import torch
import torch.nn as nn
from Attention import MHA
from FFN import FFN
from Norm import LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)

        self.mha = MHA(config)
        self.ffn = FFN(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config['n_layers'])
        ])
        self.final_norm = LayerNorm(config)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
