''' __init__ function for models

author:
    zhangsihao yang

date:
    20220911

logs:
    20220911
        add modelnet40_meshnet2_ssl_mae
    20220918
        add shapenetpart_meshnet2_ssl_mae_test
'''
from lib.options import options

if  options.get('model', None).get('name', None):

    if options.model.name == 'meshnet2_attention':
        from lib.models.mesh.meshnet2_attention import Network
    
    if options.model.name == 'MeshAE':
        from lib.models.mesh_transformer import MeshAE

    if options.model.name == 'modelnet40_meshnet2':
        from lib.models.modelnet40.meshnet2.ssl_rec_att import Network

    if options.model.name == 'modelnet40_meshnet2_ssl_rec_att':
        from lib.models.modelnet40.meshnet2.ssl_rec_att import Network

    if options.model.name == 'modelnet40_meshnet2_ssl_mae':
        from lib.models.modelnet40.meshnet2.ssl_mae import Network

    if options.model.name == 'MoCo':
        from lib.model.moco import ConditionalMoCo, DisEntMoCo, MoCo

    if options.model.name == 'pcn':
        from lib.model.occo.pcn import pcn

    if options.model.name == 'PCNEncoder':
        from lib.model.occo.pcn_util import PCNEncoder

    if options.model.params.get('base_encoder') == 'PCNEncoder':
        from lib.model.occo.pcn_util import PCNEncoder

    if options.model.name == 'PointNetCls':
        from lib.model.pointnet import PointNetCls

    if options.model.name == 'PointNetVAE':
        from lib.model.pnvae import PointNetVAE

    if options.model.name == 'pointnet_jigsaw':
        from lib.model.pointnet.jigsaw import Net

    if options.model.params.base_encoder.get('name') == 'pointnet_jigsaw_moco':
        from lib.model.pointnet.jigsaw import SimpleNet

    if options.model.name == 'pointnet_jigsaw_moco':
        from lib.model.pointnet.jigsaw import SimpleNet

    if options.model.params.base_encoder.get('name') == 'pointnet_conditional':
        from lib.model.pointnet.conditional import CNet, DENet, FCNet

    if options.model.name == 'shapenetpart_meshnet2_ssl_mae_test_train':
        from lib.models.shapenetpart.meshnet2.ssl_mae_test_.train import Network

    if options.model.name == 'shapenetpart_meshnet2_ssl_mae_test':
        from lib.models.shapenetpart.meshnet2.ssl_mae_test import Network

    if options.model.name == 'shrec16_meshnet':
        from lib.model.meshnet import MeshNet2

    if options.model.name == 'shrec16_meshnet_trans':
        from lib.model.mesh.meshnet_trans import Network

    if options.model.name == 'shrec16_spconv_trans':
        from lib.model.spconv.fcnn import Network

    if options.model.name == 'shrec16_theme':
        from lib.model.mesh.trans_unet import Network
