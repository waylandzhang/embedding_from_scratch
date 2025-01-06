"""
原始的 BERT 使用了 BooksCorpus（8 亿词）和英文维基百科（25 亿词）进行预训练。
我们这个演示学习，使用《三体》全文作为训练数据，以及自己的中文tokenizer一起，来训练一个 BERT 模型。
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import sys
import os
import random
from collections import Counter
from minbpe import BPETokenizer


# 加载我们训练的中文分词器
tokenizer = BPETokenizer()
tokenizer.load("improved_chinese_tokenizer.model")
vocab_size = len(tokenizer.vocab)
context_length = 64

corpus_files = [
    'three_body_utf8.txt',
    # '三体.txt',
    # '科幻小说.txt',
    # '人类简史.txt',
    # '天龙八部.txt',
    # '盗墓笔记.txt',
    # '经济学原理.txt',
    # '莫言中短篇小说散文选.txt',
    # '货币战争.txt',
    # '货币战争升级版.txt',
    # '马云如是说.txt'
]
# 加载corpus_files中的文本
text = ""
for file in corpus_files:
    text += open(file, "r", encoding="utf-8").read()
    
text = text.replace("\u3000", "").replace("\n", "")


class NovelDataset(Dataset):
    def __init__(self, mask_prob=0.15, max_n=5):
        self.tokenizer = tokenizer
        self.sentences = self._split_into_sentences(text)
        self.context_length = context_length
        self.mask_prob = mask_prob  # n-gram mask probability
        self.max_n = max_n  # maximum n-gram mask length

    def _split_into_sentences(self, text):
        # 使用多种标点符号分割，并过滤空句子
        seps = ["。", "！", "？", "…"]
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in seps:
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        if current.strip():  # 处理最后一个句子
            sentences.append(current.strip())
        return sentences

    def _mask_tokens(self, tokens):
        masked_tokens = []
        token_targets = []
        inverse_token_mask = []
        for token in tokens:
            if random.random() < self.mask_prob:
                r = random.random()
                if r < 0.8:  # 80% 的情况
                    masked_tokens.append(self.tokenizer.encode("[MASK]", allowed_special="all")[0])
                elif r < 0.9:  # 10% 的情况
                    masked_tokens.append(token)
                else:  # 10% 的情况
                    masked_tokens.append(random.randint(0, len(self.tokenizer.vocab) - 1))
                inverse_token_mask.append(1)  # Masked
                token_targets.append(token)  # Original token
            else:
                masked_tokens.append(token)
                inverse_token_mask.append(0)  # Not masked
                token_targets.append(-100)  # 使用 -100 表示不计算损失
        return masked_tokens, token_targets, inverse_token_mask

    def _mask_tokens_n_gram(self, tokens):
        masked_tokens = tokens.copy()
        token_targets = [-100] * len(tokens)  # 初始化为 -100 pytorch 的 CrossEntropyLoss 计算损失时会忽略 -100
        inverse_token_mask = [0] * len(tokens)  # 初始化为 0

        mask_indices = n_gram_mask(tokens, self.mask_prob, self.max_n)

        for idx in mask_indices:
            r = random.random()
            if r < 0.8:  # 80% 的情况
                masked_tokens[idx] = self.tokenizer.encode("[MASK]", allowed_special="all")[0]
            elif r < 0.9:  # 10% 的情况
                masked_tokens[idx] = tokens[idx]
            else:  # 10% 的情况
                masked_tokens[idx] = random.randint(0, len(self.tokenizer.vocab) - 1)

            inverse_token_mask[idx] = 1  # Masked 掩码token
            token_targets[idx] = tokens[idx]  # 原始token

        return masked_tokens, token_targets, inverse_token_mask

    def __len__(self):
        return len(self.sentences) - 1

    def __getitem__(self, idx):
        random.seed(idx)
        sentence_a = self.sentences[idx]
        if random.random() > 0.5:
            sentence_b = self.sentences[idx + 1]
            is_next = 1  # 下一个句子是连续的
        else:
            random_idx = random.randint(0, len(self.sentences) - 1)
            sentence_b = self.sentences[random_idx]
            is_next = 0  # 随机选择的句子作为下一个句子

        tokens_a = self.tokenizer.encode(sentence_a)
        tokens_b = self.tokenizer.encode(sentence_b)

        tokens_a, token_targets_a, inverse_token_mask_a = self._mask_tokens_n_gram(tokens_a)
        tokens_b, token_targets_b, inverse_token_mask_b = self._mask_tokens_n_gram(tokens_b)

        # 合并并截断序列
        input_ids = [self.tokenizer.encode("[CLS]", allowed_special="all")[0]] + tokens_a + \
                    [self.tokenizer.encode("[SEP]", allowed_special="all")[0]] + tokens_b + \
                    [self.tokenizer.encode("[SEP]", allowed_special="all")[0]]

        token_targets = [-100] + token_targets_a + [-100] + token_targets_b + [-100]
        inverse_token_mask = [0] + inverse_token_mask_a + [0] + inverse_token_mask_b + [0]
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        # 截断序列
        if len(input_ids) > self.context_length:
            input_ids = input_ids[:self.context_length]
            token_targets = token_targets[:self.context_length]
            inverse_token_mask = inverse_token_mask[:self.context_length]
            token_type_ids = token_type_ids[:self.context_length]

        # 计算需要的填充长度
        padding_length = self.context_length - len(input_ids)

        # 填充
        if padding_length > 0:
            input_ids += [self.tokenizer.encode("[PAD]", allowed_special="all")[0]] * padding_length
            token_targets += [-100] * padding_length
            inverse_token_mask += [0] * padding_length
            token_type_ids += [0] * padding_length

        # 创建 attention mask
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "token_targets": torch.tensor(token_targets, dtype=torch.long),
            "inverse_token_mask": torch.tensor(inverse_token_mask, dtype=torch.long),
            "nsp_target": torch.tensor(is_next, dtype=torch.long)
        }


def n_gram_mask(tokens, mask_prob=0.15, max_n=5):
    num_tokens = len(tokens)
    num_to_mask = int(num_tokens * mask_prob)
    masked_tokens = tokens.copy()
    mask_indices = set()

    while len(mask_indices) < num_to_mask:
        start = random.randint(0, num_tokens - 1)
        if start in mask_indices:
            continue

        n = random.randint(1, min(max_n, num_tokens - start))
        for i in range(start, min(start + n, num_tokens)):
            if len(mask_indices) < num_to_mask:
                mask_indices.add(i)
            else:
                break

    return list(mask_indices)
