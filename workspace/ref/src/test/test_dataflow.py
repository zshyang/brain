''' test function for  testing the data flow in network.
it will include dataset, dataloader,

function:
    f00 _dict_cuda(dict_tensor)
    f01 test_shrec_spconv()
    f02 test_shrec_ori_meshnet()
    f03 test_shrec_sum_attention()
        <-- f04 get_shre11_meshnet_dataloader
    f04 get_shre11_meshnet_dataloader(debug=False)

author:
    Zhangsihao Yang

date:
    20220605

logs:
    20220605
        create
    20220827
        f02 
    20220830
        add     f03 f04
'''
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
# from lib.dataset.mesh.shrec import SHREC16
# from lib.model.mesh.mesh_transformer import BlockTransformerEncoder
# from lib.model.mesh.trans_unet import Network
from torch.utils.data import DataLoader
from tqdm import tqdm


def test_modelnet40_meshnet2_ssl_rec_att(debug=False):
    from lib.datasets.modelnet40.meshnet2.ssl_rec_att import Dataset
    dataset = Dataset('train')
    if debug:
        for data in tqdm(dataset):
            pass

    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=dataset.__collate__
    )
    if debug:
        for data in tqdm(dataloader):
            print(data)

    from lib.models.modelnet40.meshnet2.ssl_rec_att import Network

    en_kwargs = {
        'num_faces': 1024, 'num_cls': 40, 'pool_rate': 4,
        'cfg': {
            'num_kernel': 64,
            'ConvSurface':{
                'num_samples_per_neighbor': 4,
                'rs_mode': 'Weighted', # Uniform/Weighted
                'num_kernel': 64
            },
            'MeshBlock':{
                'blocks': [3, 4, 4]
            }
        },
    }
    de_kwargs = {
        'bneck_size': 1024
    }

    network = Network(en_config=en_kwargs, de_config=de_kwargs).cuda()

    for i, data in enumerate(dataloader):
        _dict_cuda(data[0])
        x = network(**data[0])
        print(x)
        print(x.shape)
        break


def get_shre11_meshnet_dataloader(debug=False):
    ''' load shrec11 meshes using meshnet2 design
    '''
    from lib.dataset.mesh.shrec_meshnet import SHREC11
    shrec16 = SHREC11(
        data_root='/dataset/shrec16/10-10_A/',
        partition='train', augment='rotate'
    )
    if debug:
        for data in shrec16:
            print(data)

    dataloader = DataLoader(
        shrec16, batch_size=2, shuffle=True,
        collate_fn=shrec16.__collate__
    )
    if debug:
        for data in dataloader:
            print(data)
    
    return dataloader        


def test_shrec_sum_attention():
    ''' replace sum operator in meshnet2 with 
    attention operator and debug
    '''

    dataloader = get_shre11_meshnet_dataloader()

    from lib.model.mesh.meshnet2_attention import Network

    kwargs = {
        'num_faces': 500, 'num_cls': 30, 'pool_rate': 2,
        'cfg': {
            'num_kernel': 64,
            'ConvSurface':{
                'num_samples_per_neighbor': 4,
                'rs_mode': 'Weighted', # Uniform/Weighted
                'num_kernel': 64
            },
            'MeshBlock':{
                'blocks': [3, 4, 4]
            }
        },
    }

    network = Network(**kwargs).cuda()

    for i, data in enumerate(dataloader):
        _dict_cuda(data[0])
        x = network(**data[0])
        break


def _dict_cuda(dict_tensor):
        ''' this function is to move a dict of tensors onto gpu
        '''
        for key in dict_tensor:
            if type(dict_tensor[key]) is list:
                continue
            if type(dict_tensor[key]) is dict:
                for in_key in dict_tensor[key]:
                    _dict_cuda(dict_tensor[key])
            if type(dict_tensor[key]) is torch.Tensor:
                dict_tensor[key] = dict_tensor[key].cuda()


def test_shrec_ori_meshnet():
    from lib.dataset.mesh.shrec_meshnet import SHREC11
    shrec16 = SHREC11(
        data_root='/dataset/shrec16/10-10_A/',
        partition='train', augment='rotate'
    )
    for data in shrec16:
        print(data)
        break

    dataloader = DataLoader(
        shrec16, batch_size=2, shuffle=True,
        collate_fn=shrec16.__collate__
    )
    for data in dataloader:
        print(data)
        break

    from lib.model.meshnet import MeshNet2

    kwargs = {
        'num_faces': 500, 'num_cls': 30, 'pool_rate': 2,
        'cfg': {
            'num_kernel': 64,
            'ConvSurface':{
                'num_samples_per_neighbor': 4,
                'rs_mode': 'Weighted', # Uniform/Weighted
                'num_kernel': 64
            },
            'MeshBlock':{
                'blocks': [3, 4, 4]
            }
        },
    }

    network = MeshNet2(**kwargs).cuda()

    for i, data in enumerate(dataloader):
        _dict_cuda(data[0])
        x = network(**data[0])
        break


def test_shrec_meshnet():
    from lib.dataset.mesh.shrec_meshnet import SHREC11
    shrec16 = SHREC11(
        data_root='/dataset/shrec16/10-10_A/',
        partition='train', augment='rotate'
    )
    for data in shrec16:
        print(data)
        break

    dataloader = DataLoader(
        shrec16, batch_size=2, shuffle=True,
        collate_fn=shrec16.__collate__
    )
    for data in dataloader:
        print(data)
        break

    from lib.model.mesh.meshnet_trans import Network

    kwargs = {
        'num_faces': 500, 'num_cls': 30,
        'cfg': {
            'num_kernel': 64,
            'ConvSurface':{
                'num_samples_per_neighbor': 4,
                'rs_mode': 'Weighted', # Uniform/Weighted
                'num_kernel': 64
            }
        },
    }

    network = Network(**kwargs).cuda()

    for i, data in enumerate(dataloader):
        _dict_cuda(data[0])
        x = network(**data[0])
        break


def test_shrec_spconv():
    import spconv
    from lib.dataset.mesh.shrec_spconv import SHREC16
    shrec16 = SHREC16(root='/dataset/shrec16/shrec_16/', split='train')

    dataloader = DataLoader(
        shrec16, batch_size=2, shuffle=True, collate_fn=shrec16.__collate__
    )

    from lib.model.spconv.fcnn import Network

    network = Network().cuda()

    for i, data in enumerate(dataloader):
        _dict_cuda(data[0])
        x = network(**data[0])
        break


def main():
    # fix random seed for dataset rotation
    np.random.seed(2021)

    #======== dataset ========
    kwargs = {'root': '/dataset/shrec16/shrec_16/', 'split': 'train', 'nb': 16}

    shrec16 = SHREC16(**kwargs)

    #======== dataloader ========
    dataloader = DataLoader(shrec16, batch_size=2, shuffle=True, collate_fn=shrec16.collate_fn)
    print(f'the length of dataset is {len(shrec16)}')
    print(f'the length of dataloader is {len(dataloader)}')

    for i, data in enumerate(dataloader):
        print(data[0]['fms'].shape)
        break

    #======== model ========
    block_transformer_encoder_args = {'msl': 300, 'hd': 256, 'nh': 8, 'nl': 4, 'ed': 256}
    unet_args = {'in_channels': 256, 'out_channels': 16, 'nb': 16}
    mlp_kwargs = {'in_channels': 16, 'out_channels':16, 'hidden': 128}
    model = Network(block_transformer_encoder_args, unet_args, mlp_kwargs).cuda()
    _dict_cuda(data[0])
    x = model(**data[0])

    print('+++++++++++ my impelementation is here ++++++++++')


    # block_transfomer_encoder = BlockTransformerEncoder(**block_transformer_encoder_args).cuda()


    

    _dict_cuda(data[0])
    x = block_transfomer_encoder(**data[0])

    print(data[0]['spidx'].shape)


    def single_conv(in_channels, out_channels, indice_key=None):
        return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels,
                            out_channels,
                            1,
                            bias=False,
                            indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def draw_save_voxel(voxelarray):


        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(voxelarray, edgecolor='k')
        # plt.savefig('tmp.png')
        plt.show()

    conv0 = single_conv(256, 16, 'subm0').cuda()

    print(x.shape[-2:])
    x = spconv.SparseConvTensor(x, data[0]['spidx'], data[0]['spshape'], 2)

    print(x.dense().shape)

    voxelarray = (x.dense()[0]).sum(0)

    draw_save_voxel(voxelarray)


    print(x)
    print(conv0(x))


    voxelarray = (x.dense()[0]).sum(0)

    # draw_save_voxel(voxelarray)


if __name__ == '__main__':
    # main()
    # test_shrec_spconv()
    # test_shrec_meshnet()
    # test_shrec_ori_meshnet()
    # test_shrec_sum_attention()
    test_modelnet40_meshnet2_ssl_rec_att()
