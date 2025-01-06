from minbpe import BPETokenizer
from model import *
from trainer import *
from datasets_novel import *

if torch.cuda.is_available():
    torch.cuda.empty_cache()

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR.joinpath('data/bert_checkpoints')
timestamp = datetime.utcnow().timestamp()
LOG_DIR = BASE_DIR.joinpath(f'data/logs/bert_experiment_{timestamp}')

# 我们自己的分词器
tokenizer = BPETokenizer()
tokenizer.load("improved_chinese_tokenizer.model")

# 模型参数
batch_size = 4
vocab_size = len(tokenizer.vocab)
d_model = 768
n_heads = 12
head_size = d_model // n_heads
n_layers = 12
context_length = 512
dropout = 0.1
num_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
# padding_idx = tokenizer.encode("[PAD]", allowed_special="all")[0]

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

novel_model = NovelModel(bert_model, vocab_size).to(device)

# 通过PyTorch的DataLoader加载训练数据，定义在datasets_novel.py
dataset = NovelDataset(mask_prob=0.3, max_n=5)

trainer = BertTrainer(
    model=novel_model,
    dataset=dataset,
    log_dir=LOG_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    print_progress_every=20,
    print_accuracy_every=200,
    batch_size=batch_size,
    learning_rate=3e-4,
    epochs=num_epochs
)

trainer.print_summary()
trainer()


