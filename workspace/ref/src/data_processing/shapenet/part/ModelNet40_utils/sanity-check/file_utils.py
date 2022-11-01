"""
Author: Vinit V. Singh
Collection of utility function for reading file in batches
TODO: Add NoneType and other validity checks
"""

import os
import numpy as np
import subprocess

def fcount(f_path):
    """
    Get count of valid obj files

    Args:
        f_path: list of obj file paths

    Returns:
        f_count: count of valid obj files
    """
    f_count = 0
    for f in f_path:
        if os.path.exists(f):
            if os.stat(f).st_size != 0:
                f_count += 1
    return f_count

def fpath(dir_name):
    """
    Return all obj file in a directory

    Args:
        dir_name: root path to obj files

    Returns:
        f_path: list of obj files paths
    """
    f_path = []
    for root, dirs, files in os.walk(dir_name, topdown=False):
        for f in files:
            if os.path.exists(os.path.join(root, f)):
                f_path.append(os.path.join(root, f))
    return f_path

def fname(f_path):
    """
    Get unique ID for obj file in 3DFuture Dataset

    Args:
        f_path: e.g: ../3D-FUTURE-model/0a0f0cf2-3a34-4ba2-b24f-34f361c36b3e/raw_model.obj

    Returns:
        f_name: 0a0f0cf2-3a34-4ba2-b24f-34f361c36b3e
    """
    f_name = set({})
    for f in f_path:
        f = f[f.rindex('/')+1:]
        f_name.add(f)
    return f_name
