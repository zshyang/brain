''' utilty for trainers

author
    Zhangsihao Yang

date
    04/24/2022

name convention
    bs = batch size
'''
import torch


def accuracy(x, y, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)

        bs = y.size(0)

        _, pred = x.topk(
            maxk, 1, True, True
        )
        pred = pred.t()
        correct = pred.eq(
            y.view(
                1, -1
            ).expand_as(pred)
        )

        res = []
        for k in topk:
            correct_k = correct[
                :k
            ].reshape(
                -1
            ).float().sum(
                0, keepdim=True
            )

            res.append(
                correct_k.mul_(
                    100.0 / bs
                )
            )
        return res
