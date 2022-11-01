import sys
sys.path.insert(0, '/workspace')
from lib.datasets.shapenetpart.meshnet2.ssl_mae_test_.train import Dataset
import torch
from lib.models.shapenetpart.meshnet2.ssl_mae_test_.train import Network
import os
import numpy as np
import trimesh
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

ckpt_path = '/runtime/shapenetpart/meshnet2/ssl_mae_test/train/_14.pth'

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


def draw_colored_mesh(aligned_mesh, colors, save_path):
    vertices, faces = aligned_mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba_colors = np.ones((colors.shape[0], 4))
    print(rgba_colors.shape)
    rgba_colors[:, :3] = colors
    mesh.visual.face_colors = colors * 255
    save_string = trimesh.exchange.obj.export_obj(mesh)
    # print(save_string)
    with open(save_path, 'w') as save_file:
        save_file.write(save_string)


def convert_label_to_color(point_label):
    red = np.array([1, 0, 0])
    blue = np.array([0, 0, 1])

    min_index = point_label.min()
    max_index = point_label.max()

    point_label = (point_label -min_index) / (max_index - min_index)

    return point_label * red + (1 - point_label) * blue



def load_semantic_colors(filename):
    semantic_colors = {}
    with open(filename, 'r') as fin:
        for l in fin.readlines():
            semantic, r, g, b = l.rstrip().split()
            semantic_colors[semantic] = (int(r), int(g), int(b))

    return semantic_colors


def load_mesh(mesh_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces


def get_point_path(file_pattern):
    point_root = '/datasets/shapenet/part/shapenetcore_partanno_segmentation_benchmark_v0'
    point_path = os.path.join(point_root, file_pattern[0], 'points', f'{file_pattern[1]}.pts')
    return point_path


overall_acc = []
label_folder = '/datasets/shapenet/part/mesh/label'
from sklearn.metrics import jaccard_score
for data in tqdm(dataloader):
    # print(data)
    input_data, output_data = data

    pred_logit = network(input_data['x'])

    counter = 0
    for i in tqdm(range(0, pred_logit.shape[0], 1024)):
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

        # create folder 
        results_folder = os.path.join('results', split_file_pattern[1], split_file_pattern[2])
        os.makedirs(results_folder, exist_ok=True)

        colors_filename = 'semantic_colors.txt'
        semantic_colors = load_semantic_colors(filename=colors_filename)

        colors = np.zeros([len(semantic_colors), 3])
        for si, semantic in enumerate(semantic_colors):
            colors[si, :] = semantic_colors[semantic]
        colors = colors.astype('float32') / 255.0

        pred = pred_logit[i:(1024+i), :]
        pred_choice = pred.data.max(1)[1]
        gt = output_data['y'][i:(i+1024)]

        # load the obj file and save a colored one given the prediction
        mesh_root = '/datasets/shapenet/part/mesh/simplified'
        mesh_path = os.path.join(
            mesh_root, split_file_pattern[1], split_file_pattern[2],
            'simplified.obj'
        )
        tuple_mesh = load_mesh(mesh_path)
        face_colors = colors[pred_choice]
        save_path = os.path.join(results_folder, 'model.obj')
        draw_colored_mesh(tuple_mesh, face_colors, save_path)

        # copy the point to result folder
        point_root = '/datasets/shapenet/part/shapenetcore_partanno_segmentation_benchmark_v0'
        point_path = os.path.join(
            point_root, split_file_pattern[1], 'points', f'{split_file_pattern[2]}.pts'
        )
        target_path = os.path.join(results_folder, 'point.pts')
        # point_path = get_point_path(file_pattern)
        os.system(f'cp {point_path} {target_path}')

        # convert face lable to point label
        gt_point = np.ones_like(point_face_index)
        gt_point = gt[point_face_index]
        gt_point = gt_point.numpy()

        gt_point_path = os.path.join(results_folder, 'label-sem.label')
        with open(gt_point_path, 'w') as l_file:
            for label_per_point in gt_point:
                l_file.write(f'{int(label_per_point)}\n')

        # convert face predict to point predict
        pred_point = np.ones_like(point_face_index)
        pred_point = pred_choice[point_face_index]
        pred_point = pred_point.numpy()

        pred_point_path = os.path.join(results_folder, 'pred-sem.label')
        with open(pred_point_path, 'w') as l_file:
            for label_per_point in pred_point:
                l_file.write(f'{int(label_per_point)}\n')

        # compute acc and save acc
        correct = np.equal(pred_point, gt_point).sum()
        acc = correct / pred_point.shape[0] * 100.
        overall_acc.append(acc)

        # comptue iou and save iou
        iou = jaccard_score(pred_point, gt_point, average='micro')
        cat_iou_dict[cat].append(iou)

        with open(os.path.join(results_folder, 'metric.txt'), 'w') as mfile:
            mfile.write(f'acc {acc} iou {iou}')

    break
