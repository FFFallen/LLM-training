import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    # 简化版 Transformer Block
    # 构造函数接受嵌入维度、注意力头数、前馈网络隐藏层维度和dropout率
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        # 创建多头注意力层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # 创建前馈神经网络
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),    # 第一层线性变换，把每个token的维度从embed_dim映射到高维ff_hidden_dim
            nn.ReLU(),                              # 激活函数，引入非线性
            nn.Linear(ff_hidden_dim, embed_dim),    # 第二层线性变换，把维度映射回embed_dim，以便与输入相加
        )

        # post normalization
        # 层归一化(LayerNorm对最后一个维度进行归一化)
        self.norm1 = nn.LayerNorm(embed_dim)    # 用于归一化注意力输出
        self.norm2 = nn.LayerNorm(embed_dim)    # 用于归一化前馈网络输出

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, ff_hidden_dim=1024, num_layers=2, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens=50):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :]  # 只取最后一个 token 的预测
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx