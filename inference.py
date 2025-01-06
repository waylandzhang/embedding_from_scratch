from minbpe import BPETokenizer
from model import *
from trainer import *
# from datasets_novel import *

if torch.cuda.is_available():
    torch.cuda.empty_cache()

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR.joinpath('data/bert_checkpoints')

# 使用我们自己训练的分词器
tokenizer = BPETokenizer()
tokenizer.load("improved_chinese_tokenizer.model")


# 模型参数
vocab_size = len(tokenizer.vocab)
d_model = 768
n_heads = 12
head_size = d_model // n_heads
n_layers = 12
context_length = 512
dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# 加载训练好的模型
bert_model = BERTModel(
    vocab_size,
    d_model,
    n_heads,
    head_size,
    context_length,
    n_layers,
    dropout,
    device,
)

model = NovelModel(bert_model, vocab_size).to(device)
checkpoint = torch.load(CHECKPOINT_DIR.joinpath('bert_epoch1_step-1_1721348407.pt'), map_location=device)
# print(f"This model was trained for {checkpoint['epoch']} epochs")
# print(f"The final training loss was {checkpoint['loss']}")
# print(model)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 通过词嵌入矩阵层获取词向量
word_embeddings = model.bert.embedding.token_embedding.weight.data

def get_word_vector(text, tokenizer, word_embeddings):
    token_indices = tokenizer.encode(text)
    embeddings = word_embeddings[token_indices]
    mean_pooled_embedding = torch.mean(embeddings, dim=0)

    return mean_pooled_embedding

def get_sentence_vector_with_cls(sentence, tokenizer, model):
    """可选方法：使用[CLS]标记的输出作为句子向量。原BERT的下游任务实现之一。"""
    # 对句子进行编码
    input_ids = tokenizer.encode(sentence)
    input_ids = torch.tensor([input_ids]).to(device)

    # 通过模型获取输出
    with torch.no_grad():
        outputs = model.bert(input_ids)

    # 使用[CLS]标记的输出作为句子向量
    sentence_vector = outputs[0, 0, :]
    return sentence_vector

def get_sentence_vector(sentence, tokenizer, model):
    # 对句子进行编码
    input_ids = tokenizer.encode(sentence)
    input_ids = torch.tensor([input_ids]).to(device)

    # 通过模型获取输出
    with torch.no_grad():
        outputs = model.bert(input_ids)

    # 使用平均池化 Mean-pooling
    sentence_vector = outputs.mean(dim=1)  # 对所有词向量取平均
    return sentence_vector.squeeze(0)  # 移除批次维度

    # 或者使用最大池化 Max-pooling
    # sentence_vector = outputs.max(dim=1)[0]  # 取每个维度的最大值
    # return sentence_vector.squeeze(0)


def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item()

# 对比【1】通过词嵌入矩阵计算相似度
word1 = "低维展开的智子"
word2 = "宇宙的黑暗森林状态"
vector1 = get_word_vector(word1, tokenizer, word_embeddings)
vector2 = get_word_vector(word2, tokenizer, word_embeddings)
similarity = cosine_similarity(vector1, vector2)
print(f"Embedding Layer Similarity: '{word1}' and '{word2}' is: {similarity:.4f}")

# 对比【2】通过transformer最后一层的输出计算相似度
sentence1 = "低维展开的智子"
sentence2 = "宇宙的黑暗森林状态"
# sentence2 = "时间简史主要讲了什么？"
vector1 = get_sentence_vector(sentence1, tokenizer, model)
vector2 = get_sentence_vector(sentence2, tokenizer, model)
similarity = cosine_similarity(vector1, vector2)
print(f"\"{sentence1}\" vs \"{sentence2}\" \n Transformer Layer Similarity: {similarity:.4f}")


