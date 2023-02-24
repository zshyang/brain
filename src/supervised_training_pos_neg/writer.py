'''
author:
    zhangsihao yang

logs:
    20230223: file created
'''
import os

import numpy as np


class Writer(object):
    ''' a class to write logs
    '''

    def __init__(self, log_file, log_dir, **kwargs):
        os.makedirs(log_dir, exist_ok=True)
        self.open_file = open(
            os.path.join(log_dir, log_file), 'w'
        )
        self.data = {}

        self.log_dir = log_dir

    def write(self, string):
        self.open_file.write(string)
        self.open_file.flush()

    def close(self):
        self.open_file.close()

    def add_scalar(self, tag, value, epoch):
        self.write(f'epoch: {epoch:04d} / {tag}: {value:.6f}\n')

        if epoch not in self.data:
            self.data[epoch] = {}

        if tag not in self.data[epoch]:
            self.data[epoch][tag] = []

        self.data[epoch][tag].append(value)

    def __compute_average_loss__(self, epoch, epoch_data, stage):

        # two conditions to compute average loss
        if not f'{stage}_loss' in epoch_data:
            return
        if not f'{stage}_num_pass' in epoch_data:
            return

        # compute
        loss = np.array(epoch_data[f'{stage}_loss'])
        num_pass = np.array(epoch_data[f'{stage}_num_pass'])
        average_loss = (loss * num_pass).sum() / num_pass.sum()

        # write
        self.write(
            f'epoch: {epoch:04d} / average loss: {average_loss:.6f}\n'
        )

        # store
        if f'{stage}_average_loss' not in self.data[epoch]:
            self.data[epoch][f'{stage}_average_loss'] = average_loss
        else:
            raise ValueError('average loss has been computed before')

        # update
        if f'{stage}_best_average_loss' not in self.data:
            self.data[f'{stage}_best_average_loss'] = [
                average_loss, epoch]
        if average_loss < self.data[f'{stage}_best_average_loss'][0]:
            self.data[f'{stage}_best_average_loss'] = [
                average_loss, epoch]

        # write
        best_epoch = self.data[f'{stage}_best_average_loss'][1]
        best_loss = self.data[f'{stage}_best_average_loss'][0]
        self.write(
            f'best average loss: {best_loss:.6f} at epoch {best_epoch:04d}\n'
        )

    def __compute_average_accuracy__(self, epoch, epoch_data, stage):
        # two conditions to compute average accuracy
        if not f'{stage}_num_correct' in epoch_data:
            return
        if not f'{stage}_num_pass' in epoch_data:
            return

        # compute
        num_correct = np.array(epoch_data[f'{stage}_num_correct']).sum()
        num_pass = np.array(epoch_data[f'{stage}_num_pass']).sum()
        average_accuracy = num_correct / num_pass

        # write
        self.write(
            f'epoch: {epoch:04d} / average accuracy: {average_accuracy:.6f}\n'
        )

        # store
        if f'{stage}_average_accuracy' not in self.data[epoch]:
            self.data[epoch][f'{stage}_average_accuracy'] = average_accuracy
        else:
            raise ValueError('average accuracy has been computed before')

        # update
        if f'{stage}_best_average_accuracy' not in self.data:
            self.data[f'{stage}_best_average_accuracy'] = [
                average_accuracy, epoch]
        if average_accuracy > self.data[f'{stage}_best_average_accuracy'][0]:
            self.data[f'{stage}_best_average_accuracy'] = [
                average_accuracy, epoch]

        # write
        best_epoch = self.data[f'{stage}_best_average_accuracy'][1]
        best_accuracy = self.data[f'{stage}_best_average_accuracy'][0]
        self.write(
            f'best average loss: {best_accuracy:.6f} at epoch {best_epoch:04d}\n'
        )

    def summarize(self, epoch, stage):
        self.write(f'============ {stage} summary ============\n')

        epoch_data = self.data[epoch]

        self.__compute_average_loss__(epoch, epoch_data, stage)
        self.__compute_average_accuracy__(epoch, epoch_data, stage)

        self.write('============ End of Summary =============\n')

    def draw_curve(self, epochs):
        epochs = np.arange(epochs + 1)

        train_loss = [self.data[epoch]['train_average_loss']
                      for epoch in epochs]
        train_accuracy = [self.data[epoch]['train_average_accuracy']
                          for epoch in epochs]
        val_loss = [self.data[epoch]['val_average_loss'] for epoch in epochs]
        val_accuracy = [self.data[epoch]['val_average_accuracy']
                        for epoch in epochs]
        test_loss = [self.data[epoch]['test_average_loss'] for epoch in epochs]
        test_accuracy = [self.data[epoch]['test_average_accuracy']
                         for epoch in epochs]

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(epochs, train_loss, label='train loss')
        plt.plot(epochs, val_loss, label='val loss')
        plt.plot(epochs, test_loss, label='test loss')
        plt.plot(epochs, train_accuracy, label='train accuracy')
        plt.plot(epochs, val_accuracy, label='val accuracy')
        plt.plot(epochs, test_accuracy, label='test accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'loss.png'))
