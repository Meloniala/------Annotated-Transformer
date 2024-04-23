# 复制任务：输入和目标是一样的数据

import torch
from torch.optim.lr_scheduler import LambdaLR

from utils.train_util import LabelSmoothing, DummyOptimizer, DummyScheduler, rate, run_epoch
from network import subsequent_mask, make_model

DEVICE = "cuda"

class Batch:
    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = ( src!=pad ).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]     # 预测目标
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "隐藏 PAD 和未来的词"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
    
class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return sloss.data * norm, sloss
    
def data_gen(V, batch_size, nbatches, device="cuda"):
    '''
    生成一个随机的数据集
        Params:
            V: 最大值

    '''
    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1      # 1 表示序列的开始
        src = data.requires_grad_(False).clone().detach().to(device)
        tgt = data.requires_grad_(False).clone().detach().to(device)
        yield Batch(src, tgt, pad=0)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2, device=DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        )
    )

    batch_size = 80
    for _ in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20, device=DEVICE),
            model,
            SimpleLossCompute(model.generator, criterion=criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 20, device=DEVICE),
            model,
            SimpleLossCompute(model.generator, criterion=criterion),
            DummyOptimizer,
            DummyScheduler,
            mode="eval"
        )[0]

    model.eval()
    # 测试
    src = torch.LongTensor([[0, 3, 5, 7, 9 ,2, 4, 6, 8, 0]]).to(DEVICE)
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).to(DEVICE)
    print("src: ", src)
    print("pre: ", greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

if __name__ == "__main__":
    example_simple_model()