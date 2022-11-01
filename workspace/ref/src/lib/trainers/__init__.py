''' __init__ function for trainers

author:
    zhangsihao yang

date:
    20220911

logs:
    20220911
        add modelnet40_meshnet2_ssl_rec_att
    20220919
        add shapenetpart_meshnet2_ssl_mae_test_train
'''
from lib.options import options
# from lib.trainer.meshae import MeshAETrainer
# from lib.trainer.moco import MoCoTrainer
# from lib.trainer.trainer import OcCoTrainer

if options.train.name == 'SegTrainer':
    from lib.trainers.seg import SegTrainer

if options.train.name == 'classification':
    from lib.trainers.classification import Trainer

if options.train.name == 'modelnet40_meshnet2_ssl_rec_att':
    from lib.trainers.modelnet40.meshnet2.ssl_rec_att import PointNetVAETrainer

if options.train.name == 'shapenetpart_meshnet2_ssl_mae_test_train':
    from lib.trainers.shapenetpart.meshnet2.ssl_mae_test.train import Trainer
