## 2.1 注意力机制
### 2.1.1 什么是注意力机制
注意力机制是AI和深度学习领域的核心基础技术之一，旨在模仿人脑在处理大量数据时，将注意力放在关键、相关的细节上，同时过滤掉无关的信息，。通过为输入数据的不同部分动态分配权重，使得模型可以动态聚焦于当前任务最重要的特征。

传统的机器学习模型在处理长文本或复杂图片时，往往会“一视同仁”，容易导致信息过载。而注意力机制通过以下的方式改变了这一问题：
- 动态加权：模型会自动计算并赋予输入数据各部分不同的“权重”或“分数”，重要信息权重高，无关信息权重低。
- 全局捕捉：再自注意力机制（self-attention）中，模型可以分析数据内部各个元素之间的相关性，无论两者在序列中相隔多远。
文本表示从最初的通过统计学习模型进行计算的向量空间模型、语言模型，通过 Word2Vec 的单层神经网络进入到通过神经网络学习文本表示的时代。但是，从 计算机视觉（Computer Vision，CV）为起源发展起来的神经网络，其核心架构有三种：
- **前馈神经网络（Feedforward Neural Network,FNN）**：数据从输入层单向流动到输出层，无循环结构，各层之间通过全连接或特定方式传递信息。
- **卷积神经网络（Convolutional Neural Network，CNN）**，即训练参数量远小于全连接神经网络的卷积层来进行特征提取和学习。
- **循环神经网络（Recurrent Neural Network，RNN）**：，能够使用历史信息作为输入、包含环和自重复的网络。
但是RNN和CNN虽然可以捕捉到时序信息，适合序列生成的优点，自带时序的特征表示，但是却带有两个难以弥补的缺陷：
1. 序列计算可以很好的模拟时序计算，但是也限制了计算机**并行计算**的能力。由于序列的计算是依次输入、依序计算、所以导致模型的计算能力十分受到限制。
2. RNN难以捕捉到长序列的**相关关系**。在RNN的架构中，距离越远的输入关系越难以被捕捉，同时RNN需要将整个序列读入内存依次计算，所以这个情况也限制了序列的长度。虽然 LSTM 中通过门机制对此进行了一定优化，但对于较远距离相关关系的捕捉，效果仍旧难以保证。

**注意力机制的三个核心变量**：**Query**（查询值）、**Key**（键值）和 **Value**（真值）。首先就是Q和K^T的计算得到注意力分数，这个分数反映了从Q出发，对文本的每一个token应该改分布的注意力相对大小，然后通过缩放因子和`softmax`(归一化）得到注意力权重。通过将这个注意力权重和V计算，得到最后的结果就是从Q出发计算整个文本注意力得到的结果。
```python
def attention(query,key,value,dropout=None):
	# 注意力计算函数
	d_k = query.size(-1) # 获取键向量的维度，键向量的维度和值向量的维度相同
	# 计算注意力分数
	attn_scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
	# 计算权重
	attn_weights = torch.softmax(attn_scores,dim=-1)
	# 防止过拟合
	if dropout is not None:
		attn_weights = dropout(attn_weights)
	# 计算结果
	output = torch.matmul(attn_weights, value)
	
	return output,attn_weights
	
```
但是，对于目前的 `LLM`训练过程中，需要 `Casual mask`的支持，防止看到”答案“。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(query,key,value,mask=None,dropout=None):
	'''
	标准的缩放点积注意力
	args:
		q: (b,n_heads,l,d_k)
		k: (b,n_heads,l,d_k)
		v: (b,n_heads,l,d_k)
		mask: (b,1,l,l) 可选，因果掩码或者padding掩码
		dropout: nn.Dropout示例或者None
	return:
		attn_weights: (b,n_heads,l,l)
		output: (b,n_heads,l,d_v)
	'''
	d_k = query.size(-1)
	# 计算注意力分数并实现缩放
	attn_scores = torch.matmul(query,k.transpose(-2,-1)) / math.sqrt(d_k)
	# 应用掩码（mask=0的位置设为-inf,softmax之后，就会趋于0）
	if mask is not None:
		attn_scores = attn_scores.masked_fill(mask==0,torch.float('-inf'))
	# softmax归一化
	attn_weights= torch.softmax(attn_scores,dim=-1)
	# 防止过拟合问题
	if dropout is not None:
		attn_weights = F.dropout(attn_weights,p=0.1)
	# 加权聚合value
	output = torch.matmul(attn_weights,value)
	
	return output,attn_weights  # 同时返回权重，方便调试和注意力可视化
	
```

### 2.1.2自注意力
注意力机制的本质是对两段序列的元素依次进行相似度计算，寻找出一个序列的每个元素对另一个序列的每一个元素的相关度，然后基于相关度进行一个加权，即注意力分配。而这两段序列即是计算过程中的Q、K、V 的来源。
但是，在我们的实际应用中，我们往往只需要计算 Query 和 Key 之间的注意力结果，很少存在额外的真值 Value。也就是说，我们其实只需要拟合两个文本序列。​在经典的注意力机制中，Q 往往来自于一个序列，K 与 V 来自于另一个序列，都通过参数矩阵计算得到，从而可以拟合这两个序列之间的关系。例如在 Transformer 的 Decoder 结构中，Q 来自于 Decoder 的输入，K 与 V 来自于 Encoder 的输出，从而拟合了编码信息与历史信息之间的关系，便于综合这两种信息实现未来的预测。
但在 Transformer 的 Encoder 结构中，使用的是 注意力机制的变种 —— 自注意力（self-attention，自注意力）机制。所谓自注意力，即是计算本身序列中每个元素对其他元素的注意力分布，即在计算过程中，Q、K、V 都由同一个输入通过不同的参数矩阵计算得到。在 Encoder 中，Q、K、V 分别是输入对参数矩阵 Wq、Wk、Wv 做积得到，从而拟合输入语句中每一个 token 对其他所有 token 的关系。
通过自注意力机制，我们可以找到一段文本中每一个 token 与其他所有 token 的相关关系大小，从而建模文本之间的依赖关系。

### 2.1.3 掩码自注意力

掩码自注意力，即Mask Self-attention，是使用注意力掩码的自注意力机制。掩码的作用是遮蔽一些特定位置的token，模型在学习的过程中，会忽略掉被遮蔽的token。
使用注意力掩码的核心动机是让模型只能使用历史信息进行预测而不能看到未来信息。使用注意力机制的 Transformer 模型也是通过类似于 n-gram 的语言模型任务来学习的，也就是对一个文本序列，不断根据之前的 token 来预测下一个 token，直到将整个文本序列补全。
```python
# 代码生成Mask矩阵
## 创建一个上三角矩阵，用于遮蔽未来信息
## 先创建一个1 * seq_len * seq_len的矩阵
mask = torch.fill((1,seq_len,seq_len),dtype=torch.float('-inf'))
## triu 函数的功能是创建一个上三角矩阵
mask = torch.triu(mask,diagonal=1)
```
生成的Mask矩阵会是一个上三角矩阵，上三角位置的元素均为 `-inf`，其余的位置为0。
在注意力计算时，会将计算得到的注意力分数与此掩码做和，再进行Softmax操作：
```python
attn_scores = attn_scores + mask[:,:seq_len,:seq_len]
attn_scores = F.softmax(scores.float(),dim=-1).type_as(q)
```
通过做求和，上三角区域的注意力分数结果都变成了 `-inf`，而下三角区域分数不变。在做 `softmax`操作，`-inf` 的值在经过 `Softmax` 之后会被置为 0，从而忽略了上三角区域计算的注意力分数，从而实现了注意力遮蔽。

### 2.1.4 多头注意力
Transformer 使用了多头注意力机制（Multi-Head Attention），即同时对一个语料进行多次注意力计算，每次注意力计算都能拟合不同的关系，将最后的多次结果拼接起来作为最后的输出，即可更全面深入地拟合语言信息。
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
	'''多头注意力的实现'''
	def __init__(self,dim,n_heads,max_seq_len,is_casual=False):
		super().__init__()
		assert dim % n_heads == 0, "模型隐藏层维度应整除多头"
		self.head_dim = dim // n_heads
		self.n_heads = n_heads
	
		# 映射矩阵
		self.q_proj = nn.Linear(dim,self.n_heads*self.head_dim,bias=False)
		self.k_proj = nn.Linear(dim,self.n_heads*self.head_dim,bias=False)
		self.v_proj = nn.Linear(dim,self.n_heads*self.head_dim,bias=False)
		self.o_proj = nn.Linear(self.n_heads*self.head_dim,dim,bias=False)
		
		# 注意力的dropout
		self.attn_dropout = nn.Dropout(p=0.1)
		# 残差连接的dropout
		self.resid_dropout = nn.Dropout(p=0.1)
		self.is_casual = is_casual
		
		if is_casual: 
			mask = torch.full((1, 1, max_seq_len, max_seq_len), float('-inf'))   
			mask = torch.triu(mask, diagonal=1) 
			# 注册为模型的缓冲区
			self.register_buffer("mask",mask)
	
	def forward(self,x):
		# x shape = [batch_size,seq_len,hidden_dim]
		b, l, _ = x.shape
		
		# 计算投影矩阵
		# 假设dim == hidden_dim
		# [b,l,hidden_dim]*[dim,dim]^T -> [b,l,dim]
		q,k,v = self.q_proj(x),self.k_proj(x),self.v_proj(x)
		
		# 多头维度拆分
		# [b,l,dim] -> [b,l,n_heads,head_dim]->[b,n_heads,l,head_dim]
		q = q.view(b,l,self.n_heads,self.head_dim).transpose(1,2)
		k = k.view(b,l,self.n_heads,self.head_dim).transpose(1,2)
		v = v.view(b,l,self.n_heads,self.head_dim).transpose(1,2)
		
		# 注意力分数计算
		# [b,n_heads,l,head_dim]*[b,n_heads,head_dim,l] -> [b,n_heads,l,l]
		attn_scores = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(self.head_dim)
		
		# 使用掩码
		if self.is_casual:
			assert hasattr(self,'mask')
			attn_scores = attn_scores + self.mask[:,:,:l,:l]
		
		# 计算注意力权重
		# attn_weights shape: [b,n_heads,l,l]
		attn_weights = F.softmax(scores.float(),dim=-1).type_as(x)
		
		# 做dropout
		attn_weights = self.attn_dropout(attn_weights)
		
		# 计算输出
		# [b,n_heads,l,l] * [b,n_heads,l,head_dim] -> [b，n_heads,l,head_dim]
		output = torch.matmul(attn_weights,v)
		
		# 合并头
		# [b，n_heads,l,head_dim] -> [b,l,n_heads,head_dim] -> [b,l,dim]
		output = output.transpose(1,2).contiguous().view(b,l,-1)
		
		# 最终投影回残差流
		output = self.o_proj(output)
		output = self.resid_dropout(output)
		
		return output
```

## 2.2 Encoder-Decoder
上面是`Transformer`的核心——注意力机制。在 `Transformer`中，使用注意力机制的是其两个核心组件——Encoder（编码器）和Decoder(解码器)。
### 2.2.1 前馈神经网络
前馈神经网络（Feed Forward Neural Network， FNN），每一层的神经元都和上下两层的每一个神经元完全连接的网络结构。每一个 Encoder Layer 都包含一个上文讲的注意力机制和一个前馈神经网络。
```python
class MLP(nn.Module):
	'''前馈神经网络'''
	def __init__(self,dim:int,hidden_dim:int,dropout:float):
		super().__init__()
		self.l1 = nn.Linear(dim,hidden_dim,bias=False)
		self.l2 = nn.Linear(hidden_dim,dim,bias=False)
		self.dropout=nn.Dropout(dropout)
		
	def forward(self,x):
		res1 = self.l1(x)
		out1 = F.relu(res)
		res2 = self.l2(out1)
		return self.dropout(res2)
```
### 2.2.2 层归一化 
归一化核心就是为了让不同层输入的取值范围或者分布能够一致。由于深度神经网络中每一层的输入都是上一层的输出，因此再多层传递下，对网络中较高的层，之前的所有的神经层的参数变化会导致其输入的分布发生较大的改变。也就是说，随着神经网络参数的更新，各层的输出分布式不同的，且差异会随着网络的深度的增大而增大。但是，需要预测的条件分布始终式相同的，从而也就造成了预测的误差。
因此，在深度神经网络中，往往就需要归一化的操作，将每一层的输入都归一化成标准正态分布。批归一化是指一个mini-batch上进行归一化，相当于对一个batch对样本拆分出来一部分。
```python
# LayerNorm
class LayerNorm(nn.Module):
	def __init__(self,dim:int,eps:float=1e-6):
		super().__init__()
		self.l1 = nn.Parameter(torch.ones(dim))
		self.l2 = nn.Parameter(torch.zeros(dim))
		self.eps = eps
	def forward(self,x):
		mean = x.mean(-1,keepdim=True)
		std = x.std(-1,keepdim=True)
		return self.l1 * (x - mean) / (std+self.eps) + self.l2
```
```python
# RMSNorm
class RMSNorm(nn.Module):
	def __init__(self,dim:int,eps:float=1e-6):
		super().__init__()
		self.dim = dim
		self.eps = eps
		self.weights = nn.Parameter(torch.ones(dim))
	def norm(self,x):
		return x * math.rsqrt(x.pow(2).mean(-1,keeepdim=True) + self.eps)
	def forward(self,x):
		return self.weights * self.norm(x.float()).type_as(x)
```
### 2.2.4 残差连接
由于 Transformer 模型结构较复杂、层数较深，​为了避免模型退化，Transformer 采用了残差连接的思想来连接每一个子层。残差连接，即下一层的输入不仅是上一层的输出，还包括上一层的输入。残差连接允许最底层信息直接传到最高层，让高层专注于残差的学习。
$$
x=x+MultiHeadSelfAttention(LayerNorm(x))
$$
$$
output=x+FNN(LayerNorm(x))
$$
代码中实现：
```python
# 注意力计算
h = x + self.attention.forward(self.attention_norm(x))
# 经过前馈神经网络
out = h + self.feed_forward.forward(self.fnn_norm(h))
```
### 2.2.5 Encoder
 Transformer 的 Encoder。Encoder 由 N 个 Encoder Layer 组成，每一个 Encoder Layer 包括一个注意力层和一个前馈神经网络。
 ```python
 class EncoderLayer(nn.Module):
	 '''encoder层'''
	 def __init__(self,args):
		 super().__init__()
		 self.attention = MHA(args.dim,args.n_heads,args.max_seq_len,is_casual=False)
		 self.attn_norm = LayerNorm(args.dim)
		 self.ffn_norm = LayerNorm(args.dim)
		 self.fnn = MLP(args.dim,args.dim,args.dropout)
	def forward(self,x):
		norm_x = self.attn_norm(x)
		h = x + self.attention.forward(norm_x,norm_x,norm_x)
		out = h + self.fnn.forward(self.fnn_norm(h))
 ```
 ```python
 class Encoder(nn.Module):
    '''Encoder 块'''
    def __init__(self, args):
        super(Encoder, self).__init__() 
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
 ```


### 2.2.5 Decoder
Decoder 由两个注意力层和一个前馈神经网络组成。第一个注意力层是一个掩码自注意力层，即使用 Mask 的注意力计算，保证每一个 token 只能使用该 token 之前的注意力分数；第二个注意力层是一个多头注意力层，该层将使用第一个注意力层的输出作为 query，使用 Encoder 的输出作为 key 和 value，来计算注意力分数。最后，再经过前馈神经网络：
```python
class DecoderLayer(nn.Module):
	'''decoder'''
	def __init__(self,args):
		super().__init__()
		self.attn_norm_1 = LayerNorm(args.dim)
		self.mask_attention = MHA(args,is_casual=False)
		self.attn_norm_2 = LayerNorm(agrs.dim)
		self.attention = MHA(args,is_casual=False)
		self.fnn_norm = LayerNorm(args.dim)
		self.fnn = LayerNorm(args.dim, args.dim, args.dropout)
	def forward(self,x,enc_out):
		norm_x = self.attn_norm_1(x)
		x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
		norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
		
```
```python
class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super(Decoder, self).__init__() 
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)
```
## 2.3 搭建一个Transformer
### 2.3.1 `Embeddeding`层
Embedding 层其实是一个存储固定大小的词典的嵌入向量查找表。也就是说，在输入神经网络之前，我们往往会先让自然语言输入通过分词器 `tokenizer`，分词器的作用是把自然语言输入切分成 token 并转化成一个固定的 index。
```python
self.tok_embedding = nn.Embedding(args.vocab_size,args.dim)
```
### 2.3.2 位置编码
注意力机制可以实现良好的并行计算，但是其注意力的计算方式也导致了序列中的相对位置的丢失。注意力机制并不关注序列中的每一个token的位置，这无疑就是注意力机制的一大问题。因此，为使用序列顺序信息，保留序列中相对位置信息，Transformer采用了位置编码机制。
位置编码，即根据序列中的token的相对位置对其进行编码，再对位置编码加入词向量编码中。
1. 绝对位置编码：
	正余弦位置编码
$$
PE(pos,2i)=sin(pos/100002i/dmodel)
$$
$$ 
PE(pos,2i+1)=cos(pos/100002i/dmodel)
$$
```python
class SinusoidalPE(nn.Module):
	def __init__(self,seq_len,emb_dim):
		super().__init__()
		# 预计算，不参数训练(register_buffer)
		pe = torch.zeros(max_seq_len,emb_dim)
		position = torch.arange(max_seq_len).unsequenze(1).float()
		# 频率项：1 / (10000 ^ (2i / emb_dim))
		div_term = torch.exp(
			torch.arange(0,emb_dim,2).float() * (-math.log(10000.0) / emb_dim)
		)
		# 偶数用sin,奇数用cos
		pe[:, 0::2] = torch.sin(position * div_term) 
		pe[:, 1::2] = torch.cos(position * div_term) 
		
		pe = pe.unsqueeze(0) # (1, max_seq_len, emb_dim) 
		self.register_buffer('pe', pe) 
	def forward(self, x):
		b, seq_len, _ = x.shape 
		return x + self.pe[:, :seq_len, :]
```
	优点：无需训练，可以外推到训练长度之外
	缺点：固定模式，没有学习能力，实际上外推效果受限
2. 可学习的绝对位置编码
	```python
	class LearnedPE(nn.Module):
		def __init__(self,max_seq_len,emb_dim):
			super().__init__()
			# 本质上就是一个可训练的参数查找表
			self.pod_embedding  =nn.Embedding(max_seq_len,emb_dim)
		def forward(self,x):
			batch_size,seq_len,_ = x.shape
			positions = torch.arange(seq_len,device=x.device)
			return x + self.pos_embedding(positions)
	```
	优点：简单，效果好（GPT-2采用）
	缺点：超出训练长度无法外推
3. RoPE（旋转位置编码）
	优点：相对位置感知，外推性较好
	缺点：实现比较复杂，只作用于Q/K，不作用于输入嵌入
	