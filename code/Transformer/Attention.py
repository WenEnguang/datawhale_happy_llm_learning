
import torch
import torch.nn as nn
import math

class MHA(nn.Module):
    def __init__(self, config, is_causal=None):
        """
        Args:
            config: 模型配置字典
            is_causal: 是否使用因果掩码。如果为 None，则从 config['is_causal'] 读取
        """
        super().__init__()
        assert config['emb_dim'] % config['n_heads'] == 0, "Embedding dimension must be divisible by number of heads"
        self.emb_dim = config['emb_dim']
        self.n_heads = config['n_heads']
        self.head_dim = self.emb_dim // self.n_heads
        self.qkv_bias = config['qkv_bias']

        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.qkv_bias)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

        self.dropout = nn.Dropout(config['drop_rate'])
        self.resid_dropout = nn.Dropout(config['drop_rate'])

        # is_causal 参数优先，否则从 config 读取
        if is_causal is not None:
            self.is_causal = is_causal
        else:
            self.is_causal = config['is_causal']

        if self.is_causal:
            mask = torch.triu(torch.ones(config['context_length'], config['context_length'], dtype=torch.bool), diagonal=1)
            self.register_buffer('mask', mask, persistent=False)

    def forward(self, x, enc_output=None):
        """
        Args:
            x: 输入张量，shape [batch_size, seq_len, emb_dim]
            enc_output: 编码器输出，用于交叉注意力 (K/V 来源)。
                        如果为 None，则为自注意力 (Q/K/V 均来自 x)
        """
        # x shape = [batch_size, seq_len, emb_dim]
        b, l, _ = x.shape

        if enc_output is not None:
            # 交叉注意力: Q 来自 x，K/V 来自 enc_output
            _, l_enc, _ = enc_output.shape
            q = self.q_proj(x)
            k = self.k_proj(enc_output)
            v = self.v_proj(enc_output)

            # 多头维度拆分
            # [b, l, dim] -> [b, l, n_heads, head_dim] -> [b, n_heads, l, head_dim]
            q = q.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, l_enc, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, l_enc, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            # 自注意力: Q/K/V 均来自 x
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

            # 多头维度拆分
            # [b, l, dim] -> [b, l, n_heads, head_dim] -> [b, n_heads, l, head_dim]
            q = q.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 使用因果掩码 (仅自注意力时生效)
        if self.is_causal and enc_output is None:
            attn_scores = attn_scores.masked_fill(self.mask[:l, :l], float('-inf'))

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 做 dropout
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        output = torch.matmul(attn_weights, v)

        # 合并头
        output = output.transpose(1, 2).contiguous().view(b, l, -1)

        # 最终投影回残差流
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        return output
