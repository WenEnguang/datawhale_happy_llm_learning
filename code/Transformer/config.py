
# 模型参数配置
model_config = {
    'vocab_size': 151936,  # Qwen2.5 tokenizer 词表大小          
    'context_length': 128, 
    'emb_dim': 256,
    'hidden_dim': 512,            
    'drop_rate': 0.1,
    'n_layers': 6,             
    'n_heads': 4,             
    'qkv_bias': True,
    'norm_eps': 1e-5,
    'is_causal': True
}