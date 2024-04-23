import torch

from network import make_model, subsequent_mask

def inference_test():
    # 推理
    model = make_model(11, 11, 2)

    model.eval()
    src = torch.LongTensor([[i for i in range(1, 11)]])
    src_mask = torch.ones(1, 1, 10)

    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = model.decode(
            memory=memory, src_mask=src_mask, tgt=ys, tgt_mask=subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(1):
        inference_test()

run_tests()