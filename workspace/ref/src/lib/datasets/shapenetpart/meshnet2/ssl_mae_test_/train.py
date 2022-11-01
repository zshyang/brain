'''
author:
    zhangsihao yang

logs:
    20220919
        file created
'''
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm


class Dataset(data.Dataset):
    feature_folder = '/datasets/shapenet/part/mesh/feature'
    label_folder = '/datasets/shapenet/part/mesh/label'

    def _load_data(self, partition, **kwargs):
        json_file_path = os.path.join(self.feature_folder, f'{partition}.json')
        with open(json_file_path, 'r') as json_file:
            list_file_pattern = json.load(json_file)

        list_exits_file_pattern = []
        list_label_pattern = []
        list_feature = []
        counter = 0
        # load_data
        for file_pattern in tqdm(list_file_pattern):
            split_file_pattern = file_pattern.split('/')
            
            label_path = os.path.join(
                self.label_folder, split_file_pattern[1],
                split_file_pattern[2], 'label.npz'
            )

            if not os.path.exists(label_path):
                continue

            label = np.load(label_path)
            np_unique_label = np.unique(label['face_label'])
            for unique_label in np_unique_label:
                label_index = split_file_pattern[1] + str(int(unique_label))
                if not (label_index in list_label_pattern):
                    list_label_pattern.append(label_index)

            feature_path = os.path.join(
                self.feature_folder, split_file_pattern[1],
                split_file_pattern[2], 'feature.npy'
            )
            feature = np.load(feature_path)
            list_feature.append(feature)

            list_exits_file_pattern.append(file_pattern)

            counter += 1
            if kwargs.get('debug', False):
                if counter > 3:
                    break

        # create_label
        label_pattern_dict = {}

        list_label_pattern.sort()
        # self.list_label_pattern = list_label_pattern
        for i, label_pattern in enumerate(list_label_pattern):
            label_pattern_dict[label_pattern] = i
        self.label_pattern_dict = label_pattern_dict

        counter = 0
        list_label = []
        for file_pattern in tqdm(list_file_pattern):
            split_file_pattern = file_pattern.split('/')
            label_path = os.path.join(
                self.label_folder, split_file_pattern[1],
                split_file_pattern[2], 'label.npz'
            )
            if not os.path.exists(label_path):
                continue
            label = np.load(label_path)
            face_label = np.squeeze(label['face_label'])
            string_face_label = [split_file_pattern[1] + str(int(label)) for label in face_label]
            global_face_label = list(map(label_pattern_dict.get, string_face_label))
            global_face_label = np.array(global_face_label, dtype=np.int32)
            list_label.append(global_face_label)
            
            counter += 1
            if kwargs.get('debug', False):
                if counter > 3:
                    break
        
        return list_feature, list_label, list_exits_file_pattern

    def __init__(self, partition, **kwargs):
        self.partition = partition
        if partition == 'train':
            ltf, ltl, lefp = self._load_data(partition, **kwargs)
            # lvf, lvl = self._load_data('val', **kwargs)
            # self.list_feature = ltf + lvf
            # self.list_label = ltl + lvl
            self.list_feature = ltf
            self.list_label = ltl
            self.list_file_pattern = lefp
        elif partition == 'test':
            self.list_feature, self.list_label, self.list_file_pattern = self._load_data(partition, **kwargs)

    def __len__(self):
        if self.partition == 'train':
            return int(len(self.list_feature)/5)
        return len(self.list_feature)

    def __getitem__(self, i):
        return {
            'feature': self.list_feature[i],
            'label': self.list_label[i],
            'file_pattern': self.list_file_pattern[i]
        }

    @staticmethod
    def __collate__(batch: List[Dict]):
        feature = [item['feature'] for item in batch]
        label = [item['label'] for item in batch]
        file_pattern = [item['file_pattern'] for item in batch]

        feature = np.concatenate(feature, axis=0)
        label = np.concatenate(label, axis=0)
        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(label).long()

        return {'x': feature}, {'y': label, 'fp': file_pattern}
