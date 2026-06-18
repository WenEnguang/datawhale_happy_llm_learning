
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import model_config
from Attention import MHA
from Encoder import Encoder
from Decoder import Decoder
from FFN import FFN
from Norm import LayerNorm
from PE import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 必须输入词表大小和 block_size
        assert config['vocab_size'] is not None
        assert config['context_length'] is not None
        self.config = config

        # 词嵌入层
        self.embed = nn.Embedding(config['vocab_size'], config['emb_dim'])
        # 位置编码
        self.pos_enc = PositionalEncoding(config)
        # Dropout
        self.dropout = nn.Dropout(config['drop_rate'])

        # 编解码器
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # 最后的线性层，输入 emb_dim，输出是 vocab_size
        self.final_layer = nn.Linear(config['emb_dim'], config['vocab_size'])

        # 初始化权重
        self.apply(self._init_weights)

        # 查看所有的参数数量
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, target_ids=None):
        """
        Args:
            input_ids: 编码器输入 token IDs，shape [batch_size, src_seq_len]
            target_ids: 解码器目标 token IDs，shape [batch_size, tgt_seq_len]，训练时使用
        Returns:
            logits: shape [batch_size, tgt_seq_len, vocab_size]
            loss: 交叉熵损失，推理时为 None
        """
        device = input_ids.device
        b, l = input_ids.shape
        assert l <= self.config['context_length'], \
            f"输入序列长度 {l} 不能超过 context_length {self.config['context_length']}"

        # ========== 编码器 ==========
        # 词嵌入 + 位置编码 + Dropout
        token_emb = self.embed(input_ids)           # [b, l, emb_dim]
        pos_enc = self.pos_enc(token_emb)           # [b, l, emb_dim]
        x = self.dropout(pos_enc)

        # 通过编码器
        enc_output = self.encoder(x)                # [b, l, emb_dim]

        # ========== 解码器 ==========
        if target_ids is not None:
            # 训练阶段：目标序列也需要嵌入
            tgt_emb = self.embed(target_ids)        # [b, tgt_l, emb_dim]
            tgt_emb = self.pos_enc(tgt_emb)         # [b, tgt_l, emb_dim]
            tgt_emb = self.dropout(tgt_emb)

            # 通过解码器
            dec_output = self.decoder(tgt_emb, enc_output)  # [b, tgt_l, emb_dim]

            # 通过最后的线性层
            logits = self.final_layer(dec_output)   # [b, tgt_l, vocab_size]

            # 计算损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
        else:
            # 推理阶段：逐步生成
            # 简化版：只用 decoder 做单步预测
            tgt_emb = self.embed(input_ids[:, -1:])  # 取最后一个 token
            tgt_emb = self.pos_enc(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)

            dec_output = self.decoder(tgt_emb, enc_output)
            logits = self.final_layer(dec_output[:, -1, :])  # [b, vocab_size]
            loss = None

        return logits, loss


def main():
    config = model_config
    text = "我正在学习大模型"

    tokenizer = AutoTokenizer.from_pretrained(r"E:\backup_ubuntu_218\Pre_Models\Qwen2.5_0.5B")
    inputs_token = tokenizer(
        text,
        return_tensors='pt',
        max_length=config['context_length'],
        truncation=True,
        padding='max_length'
    )

    # 实例化模型
    model = Transformer(config)

    # 前向传播
    logits, loss = model(inputs_token['input_ids'])
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss}")

    predicted_ids = torch.argmax(logits, dim=-1).item()
    output = tokenizer.decode(predicted_ids)
    print(f'Output: {output}')


if __name__ == "__main__":
    print("Starting main function")
    main()
