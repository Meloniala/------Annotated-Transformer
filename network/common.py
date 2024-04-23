from copy import deepcopy

import math

import torch
import torch.nn as nn

def clones(module, N):
    '''
    将 module 复制 N 层
    '''
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    '''
    一个 Norm 层, 接在子层后, 执行标准化的作用, 将结果变换到激活函数的有效区间
    '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        # 为什么要这样初始化？
        # 这两个参数怎么保证变换后的数值在激活函数的有效区间内？

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class SublayerConnection(nn.Module):
    '''
    Add & Norm, 即 残差连接 + layernorm
    为了简化代码(?)是先进行 Norm 在进行 Add
    '''

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 这里的 sublayer 就是 Attention 层或 FFN 层，是一个函数
        return x + self.dropout(sublayer(self.norm(x)))
    
def subsequent_mask(size):
    # 得到一个包括中线的下三角全一矩阵，用于遮盖历史信息，达到 GPT “预测下一个词”的效果。MLM 的 mask 该怎么写呢？
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        # head_i=Attention(QW_i^Q, KW_i^K, VW_i^V)
        # 4 个线性层，分别接在 Q、K、V 和 Scaled Dot-Product Attention 计算结果后
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
                                                                              
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 所有头使用同一个 mask
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for lin, x in zip(self.linears, (query, key, value))
        ]   # 1,8,3,64

        x, self.attn = attention(
            query, key, value, mask, self.dropout
        )

        x = (   # 1,3,512
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        del query, key, value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "FFN"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)