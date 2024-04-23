import torch.nn as nn
import torch

from network.common import clones, SublayerConnection, LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        Params:
            x: decoder 每一层传递的中间变量 最开始是 encoder 编码的 output
            memory: 从 encoder 来的隐藏层
            tgt_mask: 用于 memory 的 mask
        '''
        m = memory
        x = self.sublayer[0](x, lambda x :self.self_attn(x, x , x, tgt_mask))
        x = self.sublayer[1](x, lambda x :self.self_attn(x, m , m, src_mask))

        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer:DecoderLayer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            # 每层的 x 是逐层迭代的，但是其他是共用的
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)