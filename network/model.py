import torch.nn as nn
from torch.nn.functional import log_softmax, pad

from copy import deepcopy

from network import Encoder, EncoderLayer, Decoder, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, Embeddings

class Generator(nn.Module):
    '''
    接在模型尾部的一个单层线性层接 sigmoid 生成预测词的概率。
    '''
    def __init__(self, d_model, vocab) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed, tgt_embed, generator:Generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator      # generator 是怎么用的？

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), self.encoder(self.src_embed(src), src_mask), src_mask, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
def make_model(
        src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=5000, device="cuda"
):
    '''
    构造模型（翻译）
        Params:
            src_vocab: 源语言词汇量大小-vocab深度
            tgt_vocab: 目标语言词汇量大小
            N: Encoder / Decoder 子单元复制层数，论文中取 6
            d_model: 模型 embedding 维度
            d_ff: FFN 层的隐藏维大小
            h: 自注意力头的数目
            dropout: dropout 比率
    '''
    attn = MultiHeadedAttention(h, d_model)
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len)
    model = EncoderDecoder(
        Encoder(EncoderLayer(size=d_model, self_attn=deepcopy(attn), feed_forward=deepcopy(ffn), dropout=dropout), N=N),
        Decoder(DecoderLayer(size=d_model, self_attn=deepcopy(attn), src_attn=deepcopy(attn), feed_forward=deepcopy(ffn), dropout=dropout), N=N),
        src_embed=nn.Sequential(Embeddings(d_model=d_model, vocab=src_vocab), deepcopy(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model=d_model, vocab=tgt_vocab), deepcopy(position)),
        generator=Generator(d_model=d_model, vocab=tgt_vocab)
    ).to(device)

    # 官方代码说这个非常重要
    # 用 Glorot / fan_avg 初始化参数，具体而言就是对每一层的输入 fan_i 和输出 fan_o ，该层的范围是[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
