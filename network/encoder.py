import torch.nn as nn

from network.common import clones, SublayerConnection, LayerNorm

class EncoderLayer(nn.Module):
    '''
    Add & Norm(Attention) + Add & Norm(FFN) 
    '''

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)    # 2 个Add & Norm
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        rst = self.sublayer[1](x, self.feed_forward)
        return rst
    

class Encoder(nn.Module):
    '''
    EncoderLayer x 
    '''

    def __init__(self, layer:EncoderLayer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        # 为什么要在栈顶加一个 layerNorm ？
        return self.norm(x)