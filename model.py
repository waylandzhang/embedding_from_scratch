"""
Embedding向量模型约等于BERT。
从Transformer模型的角度来看，BERT是一个Encoder-only的Transformer架构。
而我们所需要的Embedding向量就是Bert架构的最后一层Transformer Block的输出。
并且，它是一个Bi-directional Encoder的架构（即没有attention mask，但是有特殊标记mask）。
关于特殊标记：
主打 [CLS]、[MUSK] 和 [SEP] 来在切分的句子中添加特殊的标记，以使模型能够学习到词与词的关系，以及句子的边界。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class EmbeddingLayer(nn.Module):
    """
    BERT Embedding 层有三种属性：
        1. Token Embedding : 类似GPT中的输入样本文字部分
        2. Positional Embedding : 通过正余弦添加位置信息
        3. Segment Embedding : 样本对顺序的标注，用于区分两个句子。如：“0”代表第一个句子，“1”代表“0”接下来的下一个句子。
        将所有三个embedding做加法，得到最终给transformer encoder层的输入。
    """

    def __init__(self, d_model, device, vocab_size, context_length):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.device = device
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model).to(self.device)
        self.segment_embedding = nn.Embedding(2, self.d_model).to(self.device)
        self.position_embedding = nn.Parameter(self.create_position_encoding().to(self.device), requires_grad=False)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def create_position_encoding(self):
        position_encoding = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding

    def forward(self, idx):
        idx = idx.to(self.device)

        sentence_size = idx.size(-1)
        segment_tensor = torch.zeros_like(idx).to(self.device)
        segment_tensor[:, sentence_size // 2 + 1:] = 1
        position_embedding = self.position_embedding[:idx.size(1), :]

        x = self.token_embedding(idx) + self.segment_embedding(segment_tensor) + position_embedding

        return self.layer_norm(x)


# Define feed forward network ｜ 定义前馈网络
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        return self.ffn(x)


# Define Scaled Dot Product Attention ｜ 定义单头注意力
class Attention(nn.Module):
    def __init__(self, d_model, head_size, context_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout
        self.Wq = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wk = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wv = nn.Linear(self.d_model, self.head_size, bias=False)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v

        return output

# Define Multi-head Attention ｜ 定义多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_size, context_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout
        self.heads = nn.ModuleList(
            [Attention(self.d_model, self.head_size, self.context_length, self.dropout) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        head_outputs = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.projection_layer(head_outputs))
        return out

# Define Transformer Block ｜ 定义Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, head_size, context_length, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, head_size, context_length, dropout)
        self.ffn = FeedForwardNetwork(d_model, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, head_size, context_length, num_blocks, dropout, device):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = head_size
        self.context_length = context_length
        self.num_blocks = num_blocks
        self.device = device
        self.embedding = EmbeddingLayer(self.d_model, self.device, vocab_size, self.context_length)
        self.transformer_blocks = nn.Sequential(*(
            [TransformerBlock(self.d_model, self.num_heads, self.head_size, self.context_length, dropout) for _ in
             range(self.num_blocks)] +
            [nn.LayerNorm(self.d_model)]
        ))

    def forward(self, idx):
        x = self.embedding(idx)
        for block in self.transformer_blocks:
            x = block(x)
        return x

class NextSentencePrediction(torch.nn.Module):
    """
    BERT架构实现目的之一：NSP任务
    即判断两个句子是否是连续的。
    主要用于句子级别的分类任务。
    """
    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 2)
        """
        如果损失函数使用的 nn.NLLLoss 的化，这里需要 LogSoftmax
        如果损失函数使用的 nn.CrossEntropyLoss 的化，这里不需要 LogSoftmax，因为 nn.CrossEntropyLoss 函数内部做了
        """
        # self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x[:, 0])  # NSP任务只返回第一个token的值，即 [CLS] token
        # x = self.softmax(x)
        return x

class MaskedLanguageModel(torch.nn.Module):
    """
    BERT架构实现目的之二：MLM任务
    即从被mask的输入序列中预测原始token。
    主要用于单词级别的分类任务。
    """
    def __init__(self, hidden, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(hidden, vocab_size)
        """
        如果损失函数使用的 nn.NLLLoss 的化，这里需要 LogSoftmax
        如果损失函数使用的 nn.CrossEntropyLoss 的化，这里不需要 LogSoftmax，因为 nn.CrossEntropyLoss 函数内部做了
        """
        # self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)  # MLM任务返回所有token的值
        # x = self.softmax(x)
        return x


class NovelModel(torch.nn.Module):
    """
    NovelModel 是我们自定义的最终模型
    [ BERT Base + Next Sentence Prediction + Masked Language Modeling ]
    """
    def __init__(self, bert: BERTModel, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence_p = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        """模型输出的是两个预测值：一个是被mask掩码掉的token的预测，一个是下一句预测"""
        return self.mask_lm(x), self.next_sentence_p(x)
