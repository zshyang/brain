''' for compute acc and iou of the trained model
author:
    zhangsihao yang
logs:
    20220927
        file modified
'''
import sys
sys.path.insert(0, '/workspace')
from lib.datasets.shapenetpart.meshnet2.ssl_mae_test_.train import Dataset
import torch
from lib.models.shapenetpart.meshnet2.ssl_mae_test_.train import Network
import os
import numpy as np
dataset = Dataset(partition='test', debug=False)

print(dataset.label_pattern_dict)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, num_workers=0, 
    collate_fn=dataset.__collate__,
    drop_last=False
)

from tqdm import tqdm

print('------------')
print(len(dataset))
print('------------')


syn2cat = '/datasets/shapenet/part/shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt'

cat2syn_dict = {}
with open(syn2cat, 'r') as ofile:
    for line in ofile:
        split_line = line.split('\t')
        syn = split_line[0]
        cat = split_line[1][:-1]
        cat2syn_dict[cat] = syn
print(cat2syn_dict)

network = Network()

ckpt_path = '/runtime/shapenetpart/meshnet2/ssl_mae_test/train/_29.pth'

ckpt = torch.load(ckpt_path)


net_state = ckpt['net']

# net_state = [ for key in net_state]
from collections import OrderedDict
 
d2 = OrderedDict(
    [
        (k[len('module.'):], v) if k.startswith('module.') else (k, v) for k, v in net_state.items()
    ]
)

network.load_state_dict(d2)


# load the pretrained weights

cat_iou_dict = {}
for cat in cat2syn_dict:
    cat_iou_dict[cat2syn_dict[cat]] = []
print(cat_iou_dict)



overall_acc = []
label_folder = '/datasets/shapenet/part/mesh/label'
from sklearn.metrics import jaccard_score
for data in tqdm(dataloader):
    # print(data)
    input_data, output_data = data

    pred_logit = network(input_data['x'])

    counter = 0
    for i in range(0, pred_logit.shape[0], 1024):
        # get current cat
        file_pattern = output_data['fp'][counter]
        print(file_pattern)
        cat = cat2syn_dict[output_data['fp'][counter].split('/')[1]]
        counter += 1

        # load the lable npz
        split_file_pattern = file_pattern.split('/')
        label_path = os.path.join(
            label_folder, split_file_pattern[1],
            split_file_pattern[2], 'label.npz'
        )
        label = np.load(label_path)
        point_face_index = label['point_to_face_index']

        pred = pred_logit[i:(1024+i), :]
        pred_choice = pred.data.max(1)[1]
        gt = output_data['y'][i:(i+1024)]

        # convert face lable to point label
        gt_point = np.ones_like(point_face_index)
        gt_point = gt[point_face_index]
        gt_point = gt_point.numpy()

        # convert face predict to point predict
        pred_point = np.ones_like(point_face_index)
        pred_point = pred_choice[point_face_index]
        pred_point = pred_point.numpy()

        # compute acc and save acc
        correct = np.equal(pred_point, gt_point).sum()
        acc = correct / pred_point.shape[0] * 100.
        overall_acc.append(acc)

        # comptue iou and save iou
        iou = jaccard_score(pred_point, gt_point, average='micro')
        cat_iou_dict[cat].append(iou)


# pass

print(sum(overall_acc) / len(overall_acc))

[print(k, sum(v)/ len(v)) for k, v in cat_iou_dict.items()]

cat_iou = [sum(v)/ len(v) for k, v in cat_iou_dict.items()]
print(sum(cat_iou) / len(cat_iou), 'cat m iou')

ins_iou = [(sum(v)/ len(v), len(v)) for k, v in cat_iou_dict.items()]
total_num_ins = [num for iou, num in ins_iou]
total_num_ins = sum(total_num_ins)
ins_iou_ = sum([num * iou for iou, num in ins_iou]) / total_num_ins
print('ins iou is', ins_iou_)
