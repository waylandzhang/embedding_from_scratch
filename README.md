### 代码说明

`/embedding_from_scratch/` 是我录制的视频课《训练自己的embedding模型》对应的代码部分。

这个目录的教学内容实现了：
 - 准备《三体》小说作为数据集
 - 从零训练了一个中文BPE的 Tokenizer 分词器
 - 从零训练了一个BERT模型（即中文 Embedding 模型，并使用了上面的分词器）
 - 根据训练的BERT模型，试验了几种不同的向量生成方法

### 代码结构

 - No.1 `train_tokenier.py` 训练分词器的代码
 - No.2 `minbpe.py` Karpathy的Tokenizer训练代码（不需要改动）
 - No.3 `train.py` 调用BERT模型训练的主程序
   - `model.py` 定义BERT模型架构
   - `datasets_novel.py` 预处理小说数据集及准备DataLoader
   - `trainer.py` 定义Optimizer、Loss、训练循环等
 - No.4 `inference.py` 测试BERT模型生成的词向量


### 关于Embedding模型

Embedding模型是自然语言处理的基础，它是将文本数据转换为计算机能够理解的数字数据的一种方式。在深度学习中，Embedding模型通常是一个神经网络模型，它的输入是文本数据，输出是文本数据的向量表示。这个向量表示可以用于文本分类、文本相似度计算、文本生成等任务。

BERT是Google在2018年提出的一种Embedding模型，它是目前最先进的自然语言处理模型之一。BERT模型的特点是：预训练（Pre-training）和微调（Fine-tuning）两阶段。预训练阶段是在大规模文本数据上进行的，目的是学习文本数据的语言模型。微调阶段是在特定任务上进行的，目的是将BERT模型应用到具体的自然语言处理任务中。

现如今的大部分Embedding模型都是基于BERT模型的改进版本，比如RoBERTa、ALBERT、XLNet等。这些模型在BERT的基础上，通过改进模型架构、训练策略、数据处理等方面，进一步提升了自然语言处理的性能。

近期大家使用较多的OpenAI-text-embedding，Nomic-text-embedding，这些模型都是类BERT的架构并且改进版本（如Nomic增加了RoPE旋转位置编码），训练数据集更丰富，得到更好的效果。

所以说，简单理解为：

** Embedding模型 = BERT模型 + 改进 **

而BERT又是一个Encoder-Only的Transformer架构。所以对于我们学完GPT-2之后，再学BERT，会发现两者有很多相似之处。。

