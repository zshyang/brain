''' test the pretrain autoencoder

author:
    zhangsihao yang

logs:
    20221101    file created
'''
import argparse

import numpy as np
import torch
from dataset import Dataset
from model import Network
from options import options, update_options
from single_gpu_train import _cuda, _fix_random, parse_args
from tqdm import tqdm
import json


def _pre_train():
    train_x = []
    train_y = []
    train_info = []
    return {
        'train_x': train_x, 'train_y': train_y, 'train_info': train_info
    }


def create_label_dict():
    return {
        'AD_pos': 0,
        'MCI_neg': 1,
        'MCI_pos': 2,
        'NL_neg': 3,
        'NL_pos': 4
    }


def _update_train(dict_info, target, pred):
    dict_label = create_label_dict()
    labels = target['mesh_index']
    labels = [dict_label[label.split('/')[3]] for label in labels]
    dict_info['train_y'].extend(labels)

    inter_fea = pred[1].data.cpu().numpy().tolist()
    dict_info['train_x'].extend(inter_fea)

    dict_info['train_info'].extend(target['mesh_index'])

    print(len(dict_info['train_x']))


def _post_train(dict_info):
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.svm import SVC

    # pick the desired label
    desired_x = []
    desired_y = []
    desired_info = []
    for x, y, info in zip(
        dict_info['train_x'], dict_info['train_y'], 
        dict_info['train_info']
    ):
        if y in [3, 4]:
            desired_x.append(x)
            desired_y.append(y)
            desired_info.append(info)
    print('num train', len(desired_x))
    # do 10 folds
    acc = []
    for json_idx in range(10):
        json_file = f'/workspace/data/meta/10_fold/{json_idx:03d}.json'

        with open(json_file, 'r') as open_file:
            data = json.load(open_file)

        X_train=[]
        X_rem = []
        # x_test = []
        y_train = []
        y_rem = []
        # y_test = []
        for x, y, info in zip(
            desired_x, desired_y, desired_info
        ):
            info = info.split('/')[-1]
            info = info[:len('130_S_4990_I349561')]
            for key in data:
                if info in data[key]['train'] or info in data[key]['val']:
                    X_train.append(x)
                    y_train.append(y)
                # if 
                if info in data[key]['test']:
                    # x_test.append(x)
                    # y_test.append(y)
                    X_rem.append(x)
                    y_rem.append(y)


        # X_train, X_rem, y_train, y_rem = train_test_split(
        #     desired_x, desired_y, train_size=0.8
        # )
        # search for a best C
        # C = 0.1
        # svc = SVC(kernel="linear")
        # svc.C = C

        C_s = np.logspace(-4, 2, 10)
        svc = SVC(kernel="linear")
        scores = []
        print('=> begin to search best C')
        for C in tqdm(C_s, desc='search C'):
            svc.C = C
            this_scores = cross_val_score(
                svc, X_train, y_train,
                n_jobs=1)
            scores.append(np.mean(this_scores))
        # print(C_s)
        # print(scores)
        # get the max score
        max_score = max(scores)
        print(max_score)
        index = scores.index(max_score)
        best_c = C_s[index]
        # print('best c is ', best_c)

        svm_model_linear = SVC(
            kernel='linear', C=best_c
        ).fit(X_train, y_train)

        accuracy = svm_model_linear.score(
            X_rem, y_rem
        )
        log_str = f'The accuracy is {(accuracy * 100):.3f}'
        print(log_str)
        acc.append(accuracy)
    
    print('avg acc is', np.mean(acc), np.std(acc))

    return {}


def test():
    dict_info = _pre_train()
    for in_data, target in tqdm(
        train_dataloader, desc='gather training data'
    ):
        _cuda(in_data, target)
        pred = net(**in_data)
        _update_train(dict_info, target, pred)
        torch.cuda.empty_cache()
    dict_post_info = _post_train(dict_info)


if __name__ == '__main__':
    parse_args()

    # set up the dataset and dataloader
    _fix_random(options.seed)

    # make the dataset and dataloader
    train_op = options.data.train
    if train_op is not None:
        train_dataset = Dataset(**train_op.dataset.params)
        collate_fn = getattr(
            train_dataset, str(train_op.dataloader.collate_fn), None
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            collate_fn=collate_fn,
            batch_size=train_op.dataloader.batch_size,
            drop_last=bool(train_op.dataloader.drop_last),
            num_workers=int(train_op.dataloader.num_workers)
        )

    # set up the network
    net = Network(**options.model.params).cuda()
    # load the pretrained weights
    net.load_state_dict(torch.load(f'/runtime/300.pt')['net'])
    net.eval()

    # do the test
    test()
