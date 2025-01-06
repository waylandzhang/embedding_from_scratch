"""
trainer部分代码修改自： https://github.com/coaxsoft/pytorch_bert/blob/master/bert/trainer.py
"""
from model import *
from datasets_novel import *
import time
from datetime import datetime
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from minbpe import BPETokenizer

tokenizer = BPETokenizer()

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def lr_warmup_schedule(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ 定义一个pytorch提供的动态学习率方法 """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def percentage(batch_size: int, max_index: int, current_index: int):
    """Calculate epoch progress percentage

    Args:
        batch_size: batch size
        max_index: max index in epoch
        current_index: current index

    Returns:
        Passed percentage of dataset
    """
    batched_max = max_index // batch_size
    return round(current_index / batched_max * 100, 2)


class BertTrainer:

    def __init__(self,
                 model: BERTModel,
                 dataset,
                 log_dir: Path,
                 checkpoint_dir: Path = None,
                 print_progress_every: int = 10,
                 print_accuracy_every: int = 50,
                 batch_size: int = 4,
                 learning_rate: float = 1e-4,
                 epochs: int = 5,
                 ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.current_epoch = 0

        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.writer = SummaryWriter(str(log_dir))
        self.checkpoint_dir = checkpoint_dir

        # self.nsp_criterion = nn.BCEWithLogitsLoss().to(device)  # BCEWithLogitsLoss 适用于二分类问题
        self.nsp_criterion = nn.CrossEntropyLoss().to(device)  # CrossEntropyLoss 适用于多分类问题
        # self.mlm_criterion = nn.NLLLoss(ignore_index=0).to(device)  # NLLLoss 适用于多分类问题
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean').to(device)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        num_training_steps = len(self.loader) * epochs
        num_warmup_steps = num_training_steps * 0.05  # 整体训练步数的前5%用于warmup学习率
        self.scheduler = lr_warmup_schedule(self.optimizer, num_warmup_steps, num_training_steps)
        self._splitter_size = 30  # 打印Log时分隔符长度
        self._ds_len = len(self.dataset)
        self._batched_len = self._ds_len // self.batch_size
        self._print_every = print_progress_every
        self._accuracy_every = print_accuracy_every

    def __call__(self):
        for self.current_epoch in range(self.current_epoch, self.epochs):
            loss = self.train()
            self.save_checkpoint(self.current_epoch, step=-1, loss=loss)

    def train(self):
        print(f"Begin epoch {self.current_epoch}")
        self.model.train()
        prev = time.time()
        average_nsp_loss = 0
        average_mlm_loss = 0
        for i, value in enumerate(self.loader):
            index = i + 1
            input_ids = value["input_ids"].to(device)
            # 注意：这里的vocab_size是我们自己训练的tokenizer的vocab_size，而不是预训练BERT的vocab_size
            assert torch.all(input_ids < vocab_size), f"Input ids contain values >= vocab size ({vocab_size})"
            inverse_token_mask = value["inverse_token_mask"].to(device).bool()
            token_target = value["token_targets"].to(device)
            nsp_target = value["nsp_target"].to(device)

            self.optimizer.zero_grad()

            token, nsp = self.model(input_ids)
            token = token.masked_fill(inverse_token_mask.unsqueeze(-1), 0)

            # 计算 MLM 损失
            batch_size, seq_len = token_target.shape
            token = token.transpose(1, 2).contiguous().view(batch_size * seq_len, -1)
            token_target = token_target.view(-1)
            mask = token_target != -100
            loss_mlm = self.mlm_criterion(token[mask], token_target[mask])

            # 计算 NSP 损失
            loss_nsp = self.nsp_criterion(nsp, nsp_target)

            # 总损失
            loss = loss_mlm + loss_nsp
            average_nsp_loss += loss_nsp
            average_mlm_loss += loss_mlm
            # print(f"MLM Loss: {loss_mlm.item()}")  # 打印mlm损失 debug
            # print(f"Token predictions: {token.argmax(-1)[:5, :10]}")  # 打印前5个样本的前10个token的预测
            # print(f"Token targets: {token_target[:5, :10]}")  # 打印对应的目标值

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪，以防止梯度爆炸。可有可无。
            self.optimizer.step()
            self.scheduler.step()  # 更新学习率

            if index % self._print_every == 0:
                elapsed = time.gmtime(time.time() - prev)
                s = self.training_summary(elapsed, index, average_nsp_loss, average_mlm_loss, lr=self.optimizer.param_groups[0]['lr'])

                print(s)

                average_nsp_loss = 0
                average_mlm_loss = 0

        return loss

    def print_summary(self):
        """辅助函数，打印模型训练的一些基本信息"""
        ds_len = len(self.dataset)
        print("Model Summary\n")
        print('=' * self._splitter_size)
        print(f"Device: {device}")
        print(f"Training dataset len: {ds_len}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batched dataset len: {self._batched_len}")
        print('=' * self._splitter_size)
        print()

    def training_summary(self, elapsed, index, average_nsp_loss, average_mlm_loss, lr):
        """辅助函数，打印训练进度"""
        passed = percentage(self.batch_size, self._ds_len, index)
        global_step = self.current_epoch * len(self.loader) + index

        print_nsp_loss = average_nsp_loss / self._print_every
        print_mlm_loss = average_mlm_loss / self._print_every

        s = f"{time.strftime('%H:%M:%S', elapsed)}"
        s += f" | Epoch {self.current_epoch + 1} | {index} / {self._batched_len} ({passed}%) | " \
             f"NSP loss {print_nsp_loss:6.2f} | MLM loss {print_mlm_loss:6.2f} " \
             f"| Learning Rate {lr:.6f}"

        self.writer.add_scalar("NSP loss", print_nsp_loss, global_step=global_step)
        self.writer.add_scalar("MLM loss", print_mlm_loss, global_step=global_step)
        self.writer.add_scalar("Learning Rate", lr, global_step=global_step)

        return s

    def save_checkpoint(self, epoch, step, loss):
        """ 训练过程中保存模型检查点 checkpoint """
        if not self.checkpoint_dir:
            return

        prev = time.time()
        name = f"bert_epoch{epoch}_step{step}_{datetime.utcnow().timestamp():.0f}.pt"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.checkpoint_dir.joinpath(name))

        print()
        print('=' * self._splitter_size)
        print(f"Model saved as '{name}' for {time.time() - prev:.2f}s")
        print('=' * self._splitter_size)
        print()

    def load_checkpoint(self, path: Path):
        print('=' * self._splitter_size)
        print(f"Restoring model {path}")
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model is restored.")
        print('=' * self._splitter_size)
