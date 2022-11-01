'''
author:
    zhangsihao yang

logs:
    20220918
        add shapenetpart_meshnet2_ssl_mae_test
        update the if condition from get function to direct call
    20220919
        add 
'''
from lib.options import options

if options.data.train.dataset.get('name', None) is not None:

    if options.data.train.dataset.name == 'mesh_shapenet':
        from lib.dataset.mesh_dataset import ShapeNet, ShapeNet642

    if options.data.train.dataset.name == 'meshnet_modelnet40':
        from lib.dataset.mesh.modelnet40 import ModelNet40

    if options.data.train.dataset.name == 'modelnet_jigsaw':
        from lib.dataset.modelnet import Jigsaw, JigsawMoCo

    if options.data.train.dataset.name == 'modelnet_multi_view':
        from lib.dataset.modelnet import OcCoJigsawMoCo

    if options.data.train.dataset.name == 'modelnet40_meshnet2':
        from lib.datasets.modelnet40.meshnet2.att import Dataset
    
    if options.data.train.dataset.name == 'modelnet40_meshnet2_ssl_rec_att':
        from lib.datasets.modelnet40.meshnet2.ssl_rec_att import Dataset

    # the modelnet point cloud dataset
    if options.data.train.dataset.name == 'ModelNetDataset':
        from lib.dataset.dataset import ModelNetDataset

    if options.data.train.dataset.name == 'OcCoModelNetDataset':
        from lib.dataset.occo import OcCoModelNetDataset

    if options.data.train.dataset.name == 'occo_modelnet_moco':
        from lib.dataset.occo import DatasetMoCo

    if options.data.train.dataset.name == 'shapenetpart_meshnet2_ssl_mae':
        from lib.datasets.shapenetpart.meshnet2.ssl_mae import Dataset

    if options.data.train.dataset.name == 'shapenetpart_meshnet2_ssl_mae_test_train':
        from lib.datasets.shapenetpart.meshnet2.ssl_mae_test_.train import Dataset

    if options.data.train.dataset.name == 'shrec_meshnet':
        from lib.dataset.mesh.shrec_meshnet import SHREC11

    if options.data.train.dataset.name == 'shrec_spconv':
        from lib.dataset.mesh.shrec_spconv import SHREC16

    if options.data.train.dataset.name == 'shrec_theme':
        from lib.dataset.mesh.shrec import SHREC16

if options.data.test.dataset.get('name', None) is not None:
    if options.data.test.dataset.name == 'shapenetpart_meshnet2_ssl_mae_test':
        from lib.datasets.shapenetpart.meshnet2.ssl_mae_test import Dataset
