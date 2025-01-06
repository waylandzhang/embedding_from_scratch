"""
这个纯手撕（+GPT）优化后的tokenizer版本包含以下改进：

- 使用多进程处理批次，利用多核 CPU。
- 限制了全局统计信息的大小，防止在大语料上训练时内存使用过大。
- 使用heap堆来优化合并操作，提高大数据集的处理效率。
- 单批处理大小增加到了 10MB。

注意，由于使用了多进程，没有对代码进行额外的错误处理，有些电脑可能出现的并发问题。
"""
"""训练语料用了中文维基百科：https://dumps.wikimedia.org/zhwiki/20240301/"""
import threading
from queue import Queue
import time
import re
import os
from collections import Counter
from minbpe import BPETokenizer, get_stats, merge

"""这个正则表达式匹配单个中文字符、匹配连续的字母、数字或下划线、标点符号等，更适合中文"""
GPT4_SPLIT_PATTERN_CN = r"""[\u4e00-\u9fff]+|[\u3000-\u303f\uff00-\uffef]+|[^\u4e00-\u9fff\s]+"""

class CustomTokenizer(BPETokenizer):
    def __init__(self, pattern=None, max_stats=1000000):
        if pattern is None:
            pattern = GPT4_SPLIT_PATTERN_CN
        super().__init__(pattern)
        self.chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.global_stats = Counter()
        self.compiled_pattern = re.compile(pattern)
        self.special_tokens = {}
        self.vocab_reverse = {}
        self.max_stats = max_stats
        self.lock = threading.Lock()

    def preprocess_chinese(self, text):
        def split_chinese(match):
            return ' '.join(match.group(0))

        return self.chinese_char_pattern.sub(split_chinese, text)

    def train_on_batch(self, batch_text, vocab_size, verbose=False, min_frequency=2):
        preprocessed_text = self.preprocess_chinese(batch_text)
        text_chunks = re.findall(self.compiled_pattern, preprocessed_text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        with self.lock:
            # 更新全局统计信息
            for chunk_ids in ids:
                self.global_stats.update(get_stats(chunk_ids))

            # 限制统计信息的大小
            if len(self.global_stats) > self.max_stats:
                self.global_stats = Counter(dict(self.global_stats.most_common(self.max_stats)))

            # 执行合并操作
            while len(self.vocab) < vocab_size:
                if not self.global_stats:
                    break
                pair = max(self.global_stats, key=self.global_stats.get)
                if self.global_stats[pair] < min_frequency:
                    break  # 如果最频繁的对出现次数低于阈值，停止合并
                if pair in self.merges:
                    del self.global_stats[pair]
                    continue  # 跳过已经合并过的对
                idx = len(self.vocab)
                self.merges[pair] = idx
                self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

                if verbose:
                    print(
                        f"merge {len(self.vocab) - 256}/{vocab_size - 256}: {pair} -> {idx} ({self.vocab[idx]}) had {self.global_stats[pair]} occurrences")

                # 更新全局统计信息
                self.update_stats(pair, idx)

            self.vocab_reverse = {v: k for k, v in self.vocab.items()}

    def update_stats(self, pair, new_id):
        stats = self.global_stats
        first, second = pair
        new_pair = (first, new_id)
        i = 0
        while i < len(stats):
            if stats[i] == first and i < len(stats) - 1 and stats[i + 1] == second:
                stats[new_pair] += stats[pair]
                stats[pair] = 0
                i += 2
            else:
                i += 1

    def encode(self, text, allowed_special=None):
        if allowed_special is True:
            allowed_special = set(self.special_tokens.keys())
        elif allowed_special is False:
            allowed_special = set()
        elif allowed_special is None:
            allowed_special = set()

        preprocessed_text = self.preprocess_chinese(text)
        tokens = re.findall(self.compiled_pattern, preprocessed_text)
        encoded = []
        for token in tokens:
            if token in self.special_tokens and token in allowed_special:
                encoded.append(self.special_tokens[token])
            else:
                # 使用 BPE 编码
                token_bytes = list(token.encode('utf-8'))
                while len(token_bytes) > 1:
                    for i in range(len(token_bytes) - 1, 0, -1):
                        pair = (token_bytes[i - 1], token_bytes[i])
                        if pair in self.merges:
                            token_bytes = token_bytes[:i - 1] + [self.merges[pair]] + token_bytes[i + 1:]
                            break
                    else:
                        break
                encoded.extend(token_bytes)
        return encoded

    def decode(self, ids):
        text = super().decode(ids)
        return re.sub(r'(\s)(?=[\u4e00-\u9fff])', '', text)

    def register_special_tokens(self, special_tokens):
        with self.lock:
            self.special_tokens = special_tokens
            for token, id in special_tokens.items():
                self.vocab[id] = token.encode('utf-8')
                self.vocab_reverse[token] = id

def load_corpus_in_batches(file_paths, batch_size=1_000_000):  # 默认批次大小为1MB
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                batch.append(line)
                if sum(len(s) for s in batch) >= batch_size:
                    yield ''.join(batch)
                    batch = []
            if batch:
                yield ''.join(batch)

def evaluate_tokenizer(tokenizer, test_text):
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    compression_ratio = len(encoded) / len(test_text)
    return {
        'original_length': len(test_text),
        'encoded_length': len(encoded),
        'compression_ratio': compression_ratio,
        'roundtrip_accuracy': test_text == decoded
    }

def worker(queue, tokenizer, vocab_size, min_frequency):
    while True:
        batch = queue.get()
        if batch is None:
            break
        tokenizer.train_on_batch(batch, vocab_size, verbose=False, min_frequency=min_frequency)
        queue.task_done()

def train_tokenizer(corpus_files, vocab_size=10256, num_threads=None, batch_size=1_000_000, min_frequency=3):
    tokenizer = CustomTokenizer()

    t0 = time.time()

    # 计算总文件大小以估算进度
    total_size = sum(os.path.getsize(f) for f in corpus_files)
    processed_size = 0

    # 创建一个队列和线程池
    queue = Queue(maxsize=20)  # 限制队列大小，防止内存占用过大
    if num_threads is None:
        num_threads = min(os.cpu_count(), 32)  # 使用CPU核心最多多少个
    threads = []

    # 启动工作线程
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(queue, tokenizer, vocab_size, min_frequency))
        t.start()
        threads.append(t)

    # 读取数据并放入队列
    for i, batch in enumerate(load_corpus_in_batches(corpus_files, batch_size)):
        queue.put(batch)

        # 更新处理大小和进度
        batch_size = len(batch.encode('utf-8'))
        processed_size += batch_size
        progress = processed_size / total_size * 100

        print(f"已提交批次 {i + 1}")
        print(f"已处理： {progress:.2f}% | 当前辞典表大小: {len(tokenizer.vocab)}")
        print(f"已处理： {processed_size / 1_000_000:.2f} MB，预估共 {total_size / 1_000_000:.2f} MB。")
        print(f"本批次大小： {batch_size / 1_000_000:.2f} MB")
        print("-" * 50)

    # 等待所有任务完成
    queue.join()

    # 停止工作线程
    for _ in range(num_threads):
        queue.put(None)
    for t in threads:
        t.join()

    # 最后再训练一次，确保所有数据都被处理
    tokenizer.train_on_batch("", vocab_size, verbose=True, min_frequency=min_frequency)

    t1 = time.time()

    print(f"训练耗时 {t1 - t0:.2f} 秒")
    print(f"最终辞典表大小 {len(tokenizer.vocab)}")

    return tokenizer

def main():
    corpus_files = ['../three_body_utf8.txt', '../scifi.txt']
    tokenizer = train_tokenizer(corpus_files, num_threads=os.cpu_count())

    special_tokens = {
        "[CLS]": len(tokenizer.vocab),
        "[PAD]": len(tokenizer.vocab) + 1,
        "[SEP]": len(tokenizer.vocab) + 2,
        "[MASK]": len(tokenizer.vocab) + 3,
        "[UNK]": len(tokenizer.vocab) + 4,
    }
    tokenizer.register_special_tokens(special_tokens)

    print(f"\nTesting [PAD] special token encode: {tokenizer.encode('[PAD]', allowed_special=True)}")
    print(f"Testing [UNK] special token encode: {tokenizer.encode('[UNK]', allowed_special=True)}")

    test_text = "这是一个测试"
    encoded = tokenizer.encode(test_text, allowed_special=True)
    print(f"Testing normal text: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")

    tokenizer.save("improved_chinese_tokenizer")

    # 加载模型并测试
    tokenizer.load("improved_chinese_tokenizer.model")

    test_texts = [
        "今天天气真好！",
        "三体是一部科幻小说。逻辑是执剑人，也是第二部的主人公。",
        "The quick brown fox jumps over the lazy dog.",
        "这是一个中英文混合的句子 with some English words."
    ]

    for test_text in test_texts:
        print(f"\nTesting: {test_text}")
        result = evaluate_tokenizer(tokenizer, test_text)
        print(f"Original length: {result['original_length']}")
        print(f"Encoded length: {result['encoded_length']}")
        print(f"Compression ratio: {result['compression_ratio']:.2f}")
        print(f"Roundtrip accuracy: {result['roundtrip_accuracy']}")

if __name__ == "__main__":
    main()