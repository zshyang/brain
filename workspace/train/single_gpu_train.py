''' train the model with single gpu.

author:
    zhangsihao yang

logs:
    20221030    file created
'''
import numpy as np
import torch


def _fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for reproducibility
    torch.backends.cudnn.deterministic = True
    # for speed
    # since we are using minkowski engine, the network could
    # not be speeded up with this option in my understanding
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parse_args()
