''' model logger. log string produced from running.
Author: Zhangsihao Yang
Date: 2022-0314

lfp = log file path
'''
import os
from pprint import pformat


class ModelLogger():
    def __init__(self, opt, verbose=False):
        root = os.path.join('/runtime', opt.outf)
        os.makedirs(root, exist_ok=True)
        print(root)

        # TODO this is not reasonable
        lfp = os.path.join(
            root, f'{opt.logger.name}_out.log')
        if os.path.exists(lfp) and (opt.logger.overwrite == False):
            self.log_fout = open(lfp, 'a', encoding='utf-8')
            self.log_fout.write(pformat(opt) + '\n')
            self.log_fout.flush()
        else:
            self.log_fout = open(lfp, 'w', encoding='utf-8')
            self.log_fout.write(pformat(opt) + '\n')
            self.log_fout.flush()

    def login(self, string):
        p_string = string + '\n'
        self.log_fout.write(p_string)
        self.log_fout.flush()
