### 什么是NLP？
NLP 是 一种让计算机理解、解释和生成人类语言的技术。其核心任务是通过计算机程序来模拟人类对语言的认知和使用过程。NLP 结合了计算机科学、人工智能、语言学和心理学等多个学科的知识和技术，旨在打破人类语言和计算机语言之间的障碍，实现无缝的交流与互动。
### NLP任务
#### 1、分词
```python
import re
text = "Hello, world. This, is a test."

reeult1 = re.split(r'\s',text) # 以空格进行切割
result2 = re.split(r'(\s)',text) # 直接进行切割
result3 = re.split(r'[,.\s]',text) #  以逗号、句点和空格进行分割

result4 = re.split(r'[.,;:?!"()\']|--|\s', text)  # 以逗号、句点、分号、问号、感叹号、括号、引号和空格进行分割
result4 = [item for item in text if item.strip()] # 去除空字符串
```


#### 2、子词切分
子词切分（Subword Segmentation）是 NLP 领域中的一种常见的文本预处理技术，旨在将词汇进一步分解为更小的单位，即子词。子词切分特别适用于处理词汇稀疏问题，即当遇到罕见词或未见过的新词时，能够通过已知的子词单位来理解或生成这些词汇。子词切分在处理那些拼写复杂、合成词多的语言（如德语）或者在预训练语言模型（如BERT、GPT系列）中尤为重要。

#### 3、词性标注
词性标注（Part-of-Speech Tagging，POS Tagging）是 NLP 领域中的一项基础任务，它的目标是为文本中的每个单词分配一个词性标签，如名词、动词、形容词等。词性标注对于理解句子结构、进行句法分析、语义角色标注等高级NLP任务至关重要。通过词性标注，计算机可以更好地理解文本的含义，进而进行信息提取、情感分析、机器翻译等更复杂的处理。

#### 4、文本分类
文本分类（Text Classification）是 NLP 领域的一项核心任务，涉及到将给定的文本自动分配到一个或多个预定义的类别中。
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, model, num_labels=2, dropout_rate=0.1):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)          # 实例化Dropout层
        self.classifier = nn.Linear(
            self.model.config.hidden_size, num_labels
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state    # (B, seq_len, hidden_size)

        # 取最后一个真实token的向量
        if attention_mask is not None:
            seq_len = attention_mask.sum(dim=1) - 1     # (B,)
            batch_size = last_hidden_state.shape[0]
            last_hidden_states = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                seq_len
            ]                                            # (B, hidden_size)
        else:
            last_hidden_states = last_hidden_state[:, -1, :]  # (B, hidden_size)

        # dropout 作用在提取的向量上，不是整个序列
        last_hidden_states = self.dropout(last_hidden_states)

        # 类型转换，防止混合精度下的类型不匹配
        last_hidden_states = last_hidden_states.float()

        # 分类头
        logits = self.classifier(last_hidden_states)     # (B, num_labels)

        # 计算loss（训练时）
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits}
```

#### 5、 实体识别
实体识别（Named Entity Recognition, NER），也称为命名实体识别，是 NLP 领域的一个关键任务，旨在自动识别文本中具有特定意义的实体，并将它们分类为预定义的类别，如人名、地点、组织、日期、时间等。

#### 6、关系抽取
关系抽取（Relation Extraction）是 NLP 领域中的一项关键任务，它的目标是从文本中识别实体之间的语义关系。

#### 7、文本摘要
文本摘要（Text Summarization）是 NLP 中的一个重要任务，目的是生成一段简洁准确的摘要，来概括原文的主要内容。

#### 8、机器翻译
机器翻译（Machine Translation, MT）是 NLP 领域的一项核心任务，指使用计算机程序将一种自然语言（源语言）自动翻译成另一种自然语言（目标语言）的过程。

#### 9、自动问答
自动问答（Automatic Question Answering, QA）是 NLP 领域中的一个高级任务，旨在使计算机能够理解自然语言提出的问题，并根据给定的数据源自动提供准确的答案。

### 文本表示
文本表示的目的是将人类语言的自然形式转化为计算机可以处理的形式，也就是将文本数据数字化，使计算机能够对文本进行有效的分析和处理。
#### 词向量
向量空间模型通过将文本（包括单词、句子、段落或整个文档）转换为高维空间中的向量来实现文本的数学化表示。在这个模型中，每个维度代表一个特征项（例如，字、词、词组或短语），而向量中的每个元素值代表该特征项在文本中的权重，这种权重通过特定的计算公式（如词频TF、逆文档频率TF-IDF等）来确定，反映了特征项在文本中的重要程度。
#### 语言模型
N-gram 模型是 NLP 领域中一种基于统计的语言模型，广泛应用于语音识别、手写识别、拼写纠错、机器翻译和搜索引擎等众多任务。N-gram模型的核心思想是基于马尔可夫假设，即一个词的出现概率仅依赖于它前面的N-1个词。这里的N代表连续出现单词的数量，可以是任意正整数。
#### Word2Vec
Word2Vec是一种流行的词嵌入（Word Embedding）技术，核心思想是利用词在文本中的上下文信息来捕捉词之间的语义关系，从而使得语义相似或相关的词在向量空间中距离较近。
#### BPE
BytePair Encoding(BPE)字节对编码，此方法允许模型将预定义词汇表中的词分级为更小的词单元，甚至单个字符，从而可以处理词汇外的词汇。




#### 训练一个可用分词器
```python
# 编写tokenizer.json的大致过程
import os
import random
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
def train_tokenizer():
	'''从 JSONL 数据集训练 BPE 分词器，定义特殊 Token，保存为标准格式。'''
	# 1、读取jsonl数据集
	def read_texts(file_path):
		with open(file_path,'r',encoding='utf-8') as f:
			for line in f:
				data = json.loads(line)
	# 数据集较大，可能会出现OOM的问题，使用概率采样
	def read_texts_v2(file_path,sample_ratio=0.2,max_chars=1000,seed=42):
		'''
		sample_rate=0.2:采样率
		max_chars=1000:最大的序列长度
		'''
		random_seed = random.Random(seed)
		with open(file_path,'r',encoding='utf-8') as f:
			for line in f:
				if random_seed.random() < sample_ratio:
					data = json.loads(line)
					
	file_path = os.path.json("Datasetfile_path",'xxx.jsonl')
	
	# 2、初始化分词器
	tokenizer = Tokenizer(model.BPE())
	tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
	
	# 3、定义特殊到的token
	special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
	
	# 4、配置BPE训练器
	trainer = trainers.BpeTrainer(
		vocab_size=6400,
		show_progress=True,
		special_tokens=special_tokens,
		initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # 确保所有单字节都被包含在初始词汇表中，避免字节级 OOV。
	)
	
	# 5、训练分词器
	tests = read_tests(file_path)
	tokenizer.train_from_iterator(tests, trainer=trainer)
	
	# 6、设置解码器并验证特殊 Token
    tokenizer.decoder = decoders.ByteLevel()
    
    # 验证特殊token的ID
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2
    
    # 7、保存分词器
    tokenizer_dir = os.path.join('root_dir','tokenizer')
    os.makedir(tokenzier_dir,exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

	# 手动创建配置文件 tokenizer_config.json

    tokenizer_config = {

        "add_bos_token": False,

        "add_eos_token": False,

        "add_prefix_space": False,

        "added_tokens_decoder": {

            "0": {

                "content": "<|endoftext|>",

                "lstrip": False,

                "normalized": False,

                "rstrip": False,

                "single_word": False,

                "special": True

            },

            "1": {

                "content": "<|im_start|>",

                "lstrip": False,

                "normalized": False,

                "rstrip": False,

                "single_word": False,

                "special": True

            },

            "2": {

                "content": "<|im_end|>",

                "lstrip": False,

                "normalized": False,

                "rstrip": False,

                "single_word": False,

                "special": True

            }

        },

        "additional_special_tokens": [],

        "bos_token": "<|im_start|>",

        "clean_up_tokenization_spaces": False,

        "eos_token": "<|im_end|>",

        "legacy": True,

        "model_max_length": 32768,

        "pad_token": "<|endoftext|>",

        "sp_model_kwargs": {},

        "spaces_between_special_tokens": False,

        "tokenizer_class": "PreTrainedTokenizerFast",

        "unk_token": "<|endoftext|>",

        "chat_template": "{}"

    }

    # 保存配置文件

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(tokenizer_config, config_file, ensure_ascii=False, indent=4)
```
