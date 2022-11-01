'''
the frame of ModelTester

test()
    _prepare_test()
    if train_data_provider:
        _pre_train()
        for:
            _update_train()
        _post_train()
    _pre_test()
    _update_test()
    _post_test()

author:
    zhangsihao yang

logs:
    20220918
        file created
'''
import os

import numpy as np
from lib.testers.tester import ModelTester


class Tester(ModelTester):
    save_folder = '/datasets/shapenet/part/mesh/feature'
    def _prepare_test(self):
        self.model_manager.load_ep(
            self.opt.test.load_epoch, net=self.net
        )

    def _pre_test(self):
        pass

    def _update_test(self, target, pred):
        fea = pred
        cor_path = target['mesh_index']
        cpu_fea = fea.data.cpu().numpy()
        for fea_, cor_path_ in zip(cpu_fea, cor_path):
            save_path = os.path.join(
                self.save_folder, cor_path_, 'feature.npy'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, fea_.T)

    def _post_test(self):
        pass
