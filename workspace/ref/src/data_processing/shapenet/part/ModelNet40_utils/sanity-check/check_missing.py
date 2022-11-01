"""
Author: Vinit V. Singh
Checks for invalid decimated files
Check if no file from the orginal dataset are missed after decimation
"""
from file_utils import fpath, fname, fcount
import numpy as np

ORIGINAL = '../ModelNet40-waterlight/'
DECIMATE = '../ModelNet40-decimate-1024/'

f_path_og = fpath(ORIGINAL)
fcount_og = fcount(f_path_og)
print('Number of valid original files: {0}'.format(fcount_og))

f_path_dcmt = fpath(DECIMATE)
fcount_dcmt = fcount(f_path_dcmt)
print('Number of valid decimated files: {0}'.format(fcount_dcmt))

if fcount_og > fcount_dcmt:
    f_name_og = fname(f_path_og)
    f_name_dcmt = fname(f_path_dcmt)
    f_missing = set({})
    sets = np.array([f_name_og, f_name_dcmt])
    lrg_set = np.argmax([len(sets[0]), len(sets[1])])
    sml_set = 1 - lrg_set
    for fl in sets[lrg_set]:
        if fl not in sets[sml_set]:
            f_missing.add(fl)
    print('Number of missing files: {0}'.format(len(f_missing)))
    print('Missing files are: ')
    print(f_missing)
