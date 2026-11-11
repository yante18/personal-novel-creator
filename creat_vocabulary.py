import json
from collections import Counter
import re
import random

# 配置
CORPUS_PATH = "dp1.txt"
VOCAB_OUT = "vocab.json"
IDX2CHAR_OUT = "idx2char.json"
META_OUT = "vocab_meta.json"
MIN_FREQ = 2
SEED = 42
# 特殊标记：填充、未知、句子开始、句子结束(EOP)
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
random.seed(SEED)

def load_and_clean_corpus(file_path):
    """读取小说文本，清洗：空行、章节标题过滤"""
    with open(file_path, "r", encoding="utf-8") as f:
        all_lines = []
        for line in f:
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 跳过章节标题：第x章
            if re.match(r"^第\d+章", line):
                continue
            all_lines.append(line)
    # 直接拼接，不额外增加空格
    full_text = "".join(all_lines)
    return full_text

def build_vocab_from_text(text, min_freq=2):
    """
    基于全部文本构建字符词表
    方案A：用全部语料建词表，val集字符不会凭空变成unk
    """
    # 统计每个字符出现次数
    counter = Counter(text)
    # 过滤低频字符，只保留出现次数>=min_freq的字符
    filtered_chars = [char for char, freq in counter.items() if freq >= min_freq]
    # 按频次降序排序，高频字符id靠前
    filtered_chars.sort(key=lambda x: counter[x], reverse=True)
    vocab = {}
    # 先写入特殊token，固定id顺序
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)
    # 再写入普通字符
    for char in filtered_chars:
        vocab[char] = len(vocab)
    # 反向词表：id -> 字符，用于解码
    idx2char = {idx: char for char, idx in vocab.items()}
    # 元信息，记录词表参数和统计信息
    meta = {
        "total_chars": len(text),
        "vocab_size": len(vocab),
        "min_freq": min_freq,
        "special_tokens": SPECIAL_TOKENS,
        "char_count": dict(counter)
    }
    return vocab, idx2char, meta

def text2ids(text, vocab):
    """文本转id，未知字符映射到<unk>"""
    unk_id = vocab["<unk>"]
    ids = [vocab.get(c, unk_id) for c in text]
    return ids

def ids2text(ids, idx2char):
    """id还原文本，过滤pad标记"""
    tokens = [idx2char[i] for i in ids if idx2char[i] != "<pad>"]
    return "".join(tokens)

if __name__ == "__main__":
    print("加载并清洗语料...")
    full_text = load_and_clean_corpus(CORPUS_PATH)
    print("语料清洗完成，总字符数(去掉标题）：", len(full_text))
    print("统计词频，构建词典...")
    vocab, idx2char, meta = build_vocab_from_text(full_text, MIN_FREQ)
    print(f"词表大小：{meta['vocab_size']}")
    # 保存三份文件
    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(IDX2CHAR_OUT, "w", encoding="utf-8") as f:
        json.dump(idx2char, f, ensure_ascii=False, indent=2)
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("词典构建完成，保存到文件中...")
    # 简单测试编解码
    text = "你好，世界！"
    ids = text2ids(text, vocab)
    recover = ids2text(ids, idx2char)
    print(f"测试文本：{text}")
    print(f"转id：{ids[:10]} ...")
    print(f"还原：{recover}")

