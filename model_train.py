import json
import re
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.date import Dateset,DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_PATH = "vocab.json"
IDX2CHAR_PATH = "idx2char.json"
CORPUS_PATH = "dp1.txt"

SEQ_LEN = 128
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 20
VAL_RATIO = 0.1
SEED = 42

N_EMBD = 256
N_HEAD = 4 #
N_LAYER = 4 #
DROPOUT = 0.1

BOS ="<bos>" 
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"

random.seed(SEED)
torch.manual_seed(SEED)#

def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(IDX2CHAR_PATH, "r", encoding="utf-8") as f:
        idx2char = json.load(f)
    idx2char = {int(k): v for k, v in idx2char.items()} #
    return vocab, idx2char

def load_raw_text():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            line = line.strip()#
            if not line:
                continue
            lines.append(line)
    full_text = "".join(lines)
    return full_text

def split_train_val_by_chapter(text, val_ratio=0.1):
    """按章节划分训练集和验证集"""
    parts = re.split(r"第\d+章", text)
    chapters = []
    for i in range(1, len(parts),2):#
       title = parts[i]
       content = parts[i+1]
       chapters.append((title, content))
    random.shuffle(chapters)
    split_pos = int(len(chapters) * (1 - val_ratio))
    train_chaps = chapters[:split_pos]
    val_chaps = chapters[split_pos:]
    return "".join(train_chaps), "".join(val_chaps)

def text2ids(text, vocab):
    """将文本转为id序列"""
    unk_id = vocab[UNK]
    return [vocab.get(c, unk_id) for c in text]#

def CharSeqDataset(Dataset):
    """字符级序列数据集"""
    def __init__(self, id_list,vocab, seq_len): ##
        self.vocab = vocab
        self.seq_len = seq_len
        self.bos_id = vocab[BOS]
        self.eos_id = vocab[EOS]
        self.pad_id = vocab[PAD]
        self.samples = []
        max_content = seq_len - 2
        for start in range(0, len(id_list), max_content): #
            chunk = id_list[start:start + max_content]
            if len(chunk) == 0:
                continue
            src = [self.bos_id] + chunk + [self.eos_id]
            label = src[1:] + [self.pad_id]  # 目标是下一个字符，最后一个位置用pad填充
            if len(src) < seq_len:
                pad_num = self.seq_len - len(src)
                src += [self.pad_id] * pad_num
                label += [self.pad_id] * pad_num
            self.samples.append((
                torch.tensor(src, dtype=torch.long),
                torch.tensor(label, dtype=torch.long)
            ))  ##
    def __len__(self):##
        return len(self.samples)##

    def __getitem__(self, idx):##
        return self.samples[idx]##

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0 #
        self.n_head = n_head
        self.head_dim = n_embd // n_head ##
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) ##

    def forward(self, x,mask):
        B, T, C = x.shape  
        qkv = self.qkv_proj(x).split(C, dim=-1) #
        q, k, v = qkv

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  #
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attention_score = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  ##
        attention_score =attention_score.masked_fill(mask == 0, float("-inf"))
        attn_weight = F.softmax(attention_score, dim=-1)
        attn_weight = self.dropout(attn_weight)
        out = attn_weight @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out ##

class TransformerBlock(nn.Module): ##
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        ) ##

def forward(self, x, mask):##
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x ##

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ]) #
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # 输出层

        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len))) #

    def forward(self, idx,targets=None): ##
        B, T = idx.shape
        tok_emb = self.token_emb(idx) 
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        x = self.drop(x)

        mask = self.tril[:T,:T]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_final(x) ##

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T,C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=PAD_ID)
        return logits, loss

if __name__ == "__main__":
    vocab,idx2char = load_vocab()
    PAD_ID = vocab[PDA]
    vocab_size = len(vocab)
    print(f"词表大小{vocab_size}")

    raw_text = load_raw_text()
    train_text, val_text = split_train_val_by_chapter(raw_text, VAL_RATIO)#
    train_ids = text2ids(train_text, vocab)
    val_ids = text2ids(val_text,vocab)
    print(f"")

