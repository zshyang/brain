'''
class:
    C00 base
        00 __init__(self, root, split, **kwargs)
            <-- 01
            <-- 02
        01 create_dict_cat(root)
        02 get_cat_from_fn(fn)
        03 __len__(self)

author:
    zhangsihao yang

date:
    20220820

logs:
    20220820
        created
'''
import os
from glob import glob

from torch.utils.data import Dataset


class base(Dataset):
    def __init__(self, root, split):
        ''' init of shrec 16 mesh dataset

        args:
            root: root location of the data files
            split: 'train', or 'test' split

        name convention:
            fn = file name
            lf = list of files
        '''
        # get the list of files
        self.lf = glob(os.path.join(root, '*', split, '*.obj'))

        # create category dictionary
        self.dict_cat = self.create_dict_cat(root)

        # get the list of label index
        self.labels = []
        for fn in self.lf:
            # compute the cateogry and append
            cat = self.dict_cat[self.get_cat_from_fn(fn)]
            self.labels.append(cat)

    @staticmethod
    def create_dict_cat(root):
        # each category is projected to a number
        list_cat_path = glob(os.path.join(root, '*'))
        # convert to a list with strings of category
        list_cat = [cat_path.split('/')[-1] for cat_path in list_cat_path]
        # convert to a dictionary
        dict_cat = {cat: i for i, cat in enumerate(list_cat)}
        return dict_cat

    @staticmethod
    def get_cat_from_fn(fn):
        return fn.split('/')[-3]

    def __len__(self):
        return len(self.lf)
    