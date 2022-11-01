'''
author:
    zhangsihao yang

logs:
    20220918
        update _dict_cuda()
'''
import numpy as np
import torch
from tqdm import tqdm


class ModelTester():
    def __init__(
        self, logger, tvtdp, net, model_manager
    ):
        self.logger = logger
        self.tvtdp = tvtdp
        self.train_data_provider = tvtdp.trdp
        self.val_data_provider = tvtdp.vldp
        if tvtdp.tsdp is None:
            self.test_data_provider = tvtdp.vldp
        else:
            self.test_data_provider = tvtdp.tsdp
        self.net = net
        self.model_manager = model_manager

    def _dict_cuda(self, dict_tensor):
        ''' this function is to move 
        a dict of tensors onto gpu
        '''
        for key in dict_tensor:
            if type(dict_tensor[key]) is list:
                continue
            if type(dict_tensor[key]) is dict:
                self._dict_cuda(dict_tensor[key])
            if type(dict_tensor[key]) is torch.Tensor:
                dict_tensor[key] = dict_tensor[key].cuda()

    def _cuda(self, in_data, target):
        ''' move input data and target onto gpu
        with helper function _dict_cuda()
        '''
        self._dict_cuda(in_data)
        self._dict_cuda(target)

    def _prepare_test(self):
        ''' helper function for test
        '''
        raise NotImplementedError('_prepare_test!')

    def _pre_train(self):
        raise NotImplementedError('_pre_train!')

    def _update_train(self, target, pred):
        raise NotImplementedError('_update_train!')

    def _post_train(self):
        raise NotImplementedError('_post_train!')

    def _pre_test(self):
        raise NotImplementedError('_pre_test!')

    def _update_test(self, target, pred):
        raise NotImplementedError('_update_test!')

    def _post_test(self):
        raise NotImplementedError('_post_test!')

    def test(self, opt):

        self.opt = opt

        self._prepare_test()

        self.net.eval()

        if self.train_data_provider is not None:
            self._pre_train()
            for _ in tqdm(range(len(self.train_data_provider)), desc='gather training data'):
                in_data, target = self.train_data_provider.get_next_batch()
                self._cuda(in_data, target)
                pred = self.net(**in_data)
                self._update_train(target, pred)
                torch.cuda.empty_cache()
                # break
            self._post_train()

        # save and test
        self._pre_test()
        for _ in tqdm(range(len(self.test_data_provider)), desc='gather test data'):
            in_data, target = self.test_data_provider.get_next_batch()
            self._cuda(in_data, target)
            pred = self.net(**in_data)
            self._update_test(target, pred)
            torch.cuda.empty_cache()
            # break
        self._post_test()


class PointNetVAETester(ModelTester):
    def _load_net(self, load_epoch):
        self.model_manager.load_ep(
            load_epoch,
            net=self.net
        )

    def _prepare_test(self):
        ''' helper function for test
        '''
        self._load_net(
            self.opt.test.load_epoch
        )

    def _pre_train(self):
        self.train_x = []
        self.train_y = []

    def _update_train(self, target, pred):
        labels = target['targets'].data.cpu().numpy().tolist()
        self.train_y.extend(labels)

        inter_fea = pred[1].data.cpu().numpy().tolist()
        self.train_x.extend(inter_fea)

    def _post_train(self):
        from sklearn.model_selection import cross_val_score
        from sklearn.svm import SVC
        C_s = np.logspace(0, 3, 10)
        svc = SVC(kernel="linear")
        scores = []
        print('=> begin to search best C')
        for C in tqdm(C_s, desc='search C'):
            svc.C = C
            this_scores = cross_val_score(
                svc, self.train_x, self.train_y,
                n_jobs=1)
            scores.append(np.mean(this_scores))
        print(C_s)
        print(scores)
        # get the max score
        max_score = max(scores)
        print(max_score)
        index = scores.index(max_score)
        best_c = C_s[index]

        # fit the model again given the best C
        self.svm_model_linear = SVC(
            kernel='linear', C=best_c).fit(
                self.train_x, self.train_y)

    def _pre_test(self):
        self.test_x = []
        self.test_y = []

    def _update_test(self, target, pred):
        labels = target['targets'].data.cpu().numpy().tolist()
        self.test_y.extend(labels)

        inter_fea = pred[1].data.cpu().numpy().tolist()
        self.test_x.extend(inter_fea)

    def _post_test(self):
        accuracy = self.svm_model_linear.score(
            self.test_x, self.test_y)
        log_str = f'The accuracy is {(accuracy * 100):.3f}'
        print(log_str)
        self.logger.login(log_str)
