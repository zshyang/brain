from lib.tester.pcn_encoder_tester import PCNEncoderTester


class PointNetTester(
    PCNEncoderTester
):
    def _prepare_test(self):
        ''' helper function for test
        '''
        self.model_manager.load_ep(
            self.opt.test.load_epoch,
            net=self.net
        )
    
    def _update_train(
        self, target, pred
    ):
        labels = target['labels'].\
        data.cpu().numpy().tolist()
        self.train_y.extend(labels)

        inter_fea = pred[2].data.\
        cpu().numpy().tolist()
        self.train_x.extend(inter_fea)

    def _update_test(self, target, pred):
        labels = target['labels'].\
        data.cpu().numpy().tolist()
        self.test_y.extend(labels)

        inter_fea = pred[2].data.\
        cpu().numpy().tolist()
        self.test_x.extend(inter_fea)
