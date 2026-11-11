import json
import random
import re
from collections import Counter

CORPUS_PATH = "scrapy01/dp1.txt"
VOCAB_OUT = "vocab.json"
IDX2CHAR_OUT = "idx2char.json"
META_OUT = "vocab_meta.json"

MIN_FREQ = 2
VAL_RATIO = 0.1
SEED = 42
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

random.seed(SEED)


def load_and_clean_corpus(file_path):
    # 读取语料，逐行清洗：
    # 1) 去掉首尾空白，空行丢弃
    # 2) 丢掉 "第N章 xxx" 这种章节标题行，避免标题进词表
    # 3) 用空格把段落拼起来，段落边界也算作一个可见字符
    with open(file_path, "r", encoding="utf-8") as f:
        all_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if re.match(r"^第\d+章", line):
                continue
            all_lines.append(line)
    full_text = " ".join(all_lines)
    return full_text


def split_train_val(text, val_ratio=0.1):
    # 字符级切分：把整段文本拆成单字符列表后随机打散，按比例切开
    # 注意：这里是随机打散，会破坏上下文顺序，只适合先跑通流程
    # 后续要做小说续写，需改成按章节切分（前 90% 章训练、后 10% 章验证）
    chars = list(text)
    random.shuffle(chars)
    split_pos = int(len(chars) * (1 - val_ratio))
    train_chars = chars[:split_pos]
    val_chars = chars[split_pos:]
    train_text = "".join(train_chars)
    val_text = "".join(val_chars)
    return train_text, val_text


def build_vocab_from_text(text, min_freq=2):
    # 只在训练集文本上统计词频（验证集不能参与，否则 OOV 率虚低）
    counter = Counter(text)
    # 过滤掉出现次数小于 min_freq 的生僻字，它们统一落到 <unk>
    filtered_chars = [char for char, freq in counter.items() if freq >= min_freq]
    # 按频次降序排列，高频字排前面，id 更小
    filtered_chars.sort(key=lambda x: counter[x], reverse=True)

    # 先放特殊 token，顺序固定，保证每次运行 id 一致
    vocab = {}
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)
    # 再按频次把字符依次编码
    for char in filtered_chars:
        vocab[char] = len(vocab)

    # 反向表：id -> 字符，生成文本时解码用
    idx2char = {idx: char for char, idx in vocab.items()}
    # 元信息落盘，方便事后核对语料规模、词表大小和字符频次
    meta = {
        "total_train_chars": len(text),
        "vocab_size": len(vocab),
        "min_freq": min_freq,
        "special_tokens": SPECIAL_TOKENS,
        "char_count": dict(counter),
    }
    return vocab, idx2char, meta


def text2ids(text, vocab):
    # 文本转 id；词表里没有的字符统一映射到 <unk>
    unk_id = vocab["<unk>"]
    ids = [vocab.get(c, unk_id) for c in text]
    return ids


def ids2text(ids, idx2char):
    # id 转回文本，跳过 <pad>，避免还原结果里出现填充符
    tokens = [idx2char[i] for i in ids if idx2char[i] != "<pad>"]
    return "".join(tokens)


if __name__ == "__main__":
    print("加载并清洗语料...")
    full_text = load_and_clean_corpus(CORPUS_PATH)
    print("语料清洗完成，总字符数(去掉标题)：", len(full_text))

    print("分割训练集和验证集...")
    train_text, val_text = split_train_val(full_text, VAL_RATIO)
    print("训练集字符数：", len(train_text))
    print("验证集字符数：", len(val_text))

    print("统计词频，构建词典...")
    vocab, idx2char, meta = build_vocab_from_text(train_text, MIN_FREQ)
    print(f"词表大小：{meta['vocab_size']}")

    # 三个 json 都按 utf-8 存储；ensure_ascii=False 让中文原样可读
    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(IDX2CHAR_OUT, "w", encoding="utf-8") as f:
        json.dump(idx2char, f, ensure_ascii=False, indent=2)
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("词典构建完成，保存到文件中...")

    # 自测：编码后再解码，检查是否能原样还原
    text = "你好，世界！"
    ids = text2ids(text, vocab)
    recover = ids2text(ids, idx2char)
    print(f"测试文本：{text}")
    print(f"转id：{ids[:10]} ...")
    print(f"还原：{recover}")
