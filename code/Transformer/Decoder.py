
import torch.nn as nn

from Attention import MHA
from Norm import LayerNorm
from FFN import FFN

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 masked attention、cross attention、MLP 之前
        self.attn1_norm = LayerNorm(config)
        self.attn2_norm = LayerNorm(config)
        self.ffn_norm = LayerNorm(config)

        # 因果自注意力 (masked self-attention)
        self.masked_mha = MHA(config, is_causal=True)
        # 交叉注意力 (cross-attention)，不启用因果掩码
        self.cross_mha = MHA(config, is_causal=False)
        self.ffn = FFN(config)

    def forward(self, x, enc_output):
        # x: decoder 输入，enc_output: encoder 输出
        # 1. 因果自注意力
        x = x + self.masked_mha(self.attn1_norm(x))
        # 2. 交叉注意力 (Q 来自 x，K/V 来自 enc_output)
        x = x + self.cross_mha(self.attn2_norm(x), enc_output)
        # 3. 前馈网络
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config['n_layers'])
        ])
        self.final_norm = LayerNorm(config)

    def forward(self, x, enc_output):
        for layer in self.layers:
            x = layer(x, enc_output)
        return self.final_norm(x)
