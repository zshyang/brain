'''
Zhangsihao Yang
04/09/2022

pcn encoder and use svm to test the test performance.

kp = key prefix

'''

import datetime
import json
import os

import numpy as np
import torch
from lib.tester.tester import ModelTester
from sklearn import metrics, svm


class PCNEncoderTester(ModelTester):
    def _prepare_test(self):
 
        # prepare for ckpt path
        ep = self.opt.test.load_epoch
        root = root = os.path.join(
            '/runtime', self.opt.outf
        )
        prefix = self.opt.manager.ckpt_prefix

        # get ckpt path and check if it exists
        ckpt_path = os.path.join(
            root, f'{prefix}_{ep}.pth'
        )
        if not self.model_manager._search_file(ckpt_path):
            raise Exception('ckpt not found')
        
        # load
        ckpt = torch.load(
            ckpt_path, map_location='cpu'
        )

        # modify state dict for single network
        state_dict = ckpt['net']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q.0.'):
                # remove prefix
                kp = k[len("module.encoder_q.0."):]
                kp = 'module.' + kp
                state_dict[kp] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        # load to self.net
        msg = self.net.load_state_dict(
            state_dict, strict=False
        )
        print(f'=> {msg}')

    def _pre_train(self):
        self.train_x = []
        self.train_y = []

    def _update_train(
        self, target, pred
    ):
        labels = target[
            'labels'
        ].data
        labels = labels.cpu().numpy().tolist()
        self.train_y.extend(labels)

        inter_fea = pred.data.cpu().numpy().tolist()
        self.train_x.extend(inter_fea)

    def _post_train(self):
        # save the gathered feature
        # save folder
        root = os.path.join(
            '/runtime', self.opt.outf
        )
        txp = os.path.join(
            root, 'train_x.json'
        )
        typ =  os.path.join(
            root, 'train_y.json'
        )
        # with open(txp, 'w') as of:
        #     json.dump(
        #         self.train_x, of
        #     )
        # with open(typ, 'w') as of:
        #     json.dump(
        #         self.train_y, of
        #     )

        from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                             cross_val_score)
        from sklearn.svm import SVC

        print(
            'Now we use Grid Search '
            'to opt the parameters '
            'for SVM Linear kernel'
        )
        # [1e-1, 5e-1, 1e0, ..., 5e1]
        # C_range = np.outer(
        #     np.logspace(-2, 2, 5), 
        #     np.array([1, 5])
        # ).flatten()
        C_range = np.array([10.])
        print(f'c range is {C_range}')
        parameters = {
            'kernel': ['linear'], 
            'C': C_range
        }

        svm_clsf = svm.SVC()
        self.grid_clsf = GridSearchCV(
            estimator=svm_clsf, 
            param_grid=parameters, 
            n_jobs=5, verbose=1
        )

        start_time = datetime.datetime.now()
        print(f'Start Param Search at {str(start_time)}')
        self.grid_clsf.fit(
            self.train_x, 
            self.train_y
        )
        print(
            'Elapsed time, param searching '
            f'{str(datetime.datetime.now() - start_time)}'
        )
        sorted(self.grid_clsf.cv_results_.keys())

    def _pre_test(self):
        self.test_x = []
        self.test_y = []

    def _update_test(self, target, pred):
        labels = target['labels'].data
        labels = labels.cpu().numpy().tolist()
        self.test_y.extend(labels)

        inter_fea = pred.data.cpu().numpy().tolist()
        self.test_x.extend(inter_fea)

    def _post_test(self):
        y_pred = self.grid_clsf.best_estimator_.predict(
            self.test_x
        )
        print("\n\n")
        print("="*37)
        print(
            'Best Params via Grid Search Cross Validation '
            'on Train Split is: '
            f'{self.grid_clsf.best_params_}'
        )
        print(
            'Best Model Accuracy on Test Dataset:'
            f'{metrics.accuracy_score(self.test_y, y_pred)}'
        )
