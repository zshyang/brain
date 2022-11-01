from __future__ import print_function

import json
import os
import os.path
import sys
from glob import glob

import h5py
import numpy as np
import torch
import torch.utils.data as data
from lib.external.python_plyfile.plyfile import PlyData
# from tqdm import tqdm


def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))


def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as open_file:
        data = json.load(open_file)
    return data


def load_ply(file_name):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']

    points = np.vstack([points['x'], points['y'], points['z']]).T

    return points


class ShapeNetPointCloud(data.Dataset):
    ''' dataset for shapenet point cloud
    '''
    def __init__(
        self, center, noise_da, noise_level, rot_da, scale,
        split):
        self.center = center
        self.noise_da = noise_da
        self.noise_level = noise_level
        self.rot_da = rot_da
        self.scale = scale
        self.split = split

        if self.split == 'train':
            json_file = '/dataset/split_meta/list_train.json'
        elif self.split == 'val':
            json_file = '/dataset/split_meta/list_val.json'
        else:
            sys.exit('No such status!')

        self.list_file = load_json(json_file)

        self.snc_synth_id_to_category = {
            '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
            '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
            '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
            '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
            '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
            '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
            '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
            '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
            '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
            '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
            '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
            '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
            '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
            '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
            '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
            '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
            '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
            '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
            '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
        }

        i = 0
        self.synth_id_to_index = {}
        for key in self.snc_synth_id_to_category:
            self.synth_id_to_index[key] = i
            i = i + 1

    def __len__(self):
        return len(self.list_file)
        # return 60

    def __getitem__(self, index):
        file_name = self.list_file[index]
        file_name = file_name.split('/')
        synth_id = file_name[2]
        file_name.insert(2, 'data')
        file_name = '/'.join(file_name)
        point_set = load_ply(file_name)

        if self.center:
            point_set = point_set - np.expand_dims(
                np.mean(point_set, axis=0), 0)  # center
        
        if self.scale:
            dist = np.max(np.sqrt(
                np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist  # scale

        # store the point set that is not augmentated
        unaug_pc = point_set

        # rotation data augmentation
        if self.rot_da:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(
                rotation_matrix)  # random rotation

        # noise data augmentation
        if self.noise_da:
            point_set += np.random.normal(
                0, self.noise_level, 
                size=point_set.shape)  # random jitter

        label = self.synth_id_to_index[synth_id]

        point_set = torch.from_numpy(
            point_set.astype(np.float32))
        label = torch.from_numpy(
            np.array([label]).astype(np.int64))
        unaug_pc = torch.from_numpy(
            unaug_pc.astype(np.float32))

        return point_set, label, unaug_pc


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def get_h5_from_json(json_filename, split):
    if split == 'train':
        return json_filename[0:-15] + json_filename[-14] + '.h5'
    else:
        return json_filename[0:-15] + json_filename[-14] + '.h5'


class ModelNetDataset(data.Dataset):
    def __init__(self, **kwargs):
        self.center = kwargs.get('center')
        self.debug = kwargs.get('debug')
        self.noise_da = kwargs.get('noise_da')
        self.noise_level = kwargs.get('noise_level')
        self.root = kwargs.get('root')
        self.rot_da = kwargs.get('rot_da')
        self.scale = kwargs.get('scale')
        self.split = kwargs.get('split')
        if self.split == 'val':
            self.split = 'test'

        self.fns = []
        self.file_index = []

        self.list_json = glob(
            os.path.join(
                self.root, 
                f'*{self.split}*.json'
            )
        )
        for i, json_file in enumerate(self.list_json):
            with open(
                json_file, 'r', encoding='utf-8'
            ) as open_file:
                fn = json.load(open_file)
                self.fns.extend(fn)
                self.file_index.extend(
                    [[i, j] for j in range(len(fn))])

        self.cat = {}
        cat_file = os.path.join(
            self.root, 'shape_names.txt')
        with open(cat_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                self.cat[line.strip()] = i
        print(self.cat)
        self.classes = list(self.cat.keys())

        self.data = []
        for i, json_file in enumerate(self.list_json):
            h5fn = get_h5_from_json(json_file, self.split)
            pc, label = load_h5(h5fn)
            self.data.append(pc)

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        json_index = self.file_index[index][0]
        inside_index = self.file_index[index][1]
        point_set = self.data[json_index][inside_index]

        if self.center:
            point_set = point_set - np.expand_dims(
                np.mean(point_set, axis=0), 0)  # center
        if self.scale:
            dist = np.max(np.sqrt(
                np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist  # scale

        if self.rot_da:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
        if self.noise_da:
            point_set += np.random.normal(0, self.noise_level, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        point_set = point_set.transpose(0, 1)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        if self.debug:
            return 248
        else:
            return len(self.fns)


if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])
