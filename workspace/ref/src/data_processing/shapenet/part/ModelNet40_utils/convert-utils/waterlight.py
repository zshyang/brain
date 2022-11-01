"""
Author: Vinit V. Singh
Wrapper around https://github.com/hjwdzh/Manifold to convert triangular mesh
to waterlight mesh
"""

import os
import subprocess

#[1]
manifold = '/media/jakep/Elements/Mesh/Decimation/Manifold/build/manifold '

src = '../ModelNet40-waterlight/'
dst = '../ModelNet40-waterlight/'

for root, dirs, files in os.walk(src, topdown = False):
    for name in files:
        if name.endswith('.obj'):
            o = root.replace(src, dst)
            if not os.path.exists(o):
                os.makedirs(o)
            i = root + os.sep + name
            o = o + os.sep + name
            cmd = manifold + i + ' ' + o + ' 20000 '
            print(cmd)
            subprocess.call(cmd, shell=True)


#References:
#1. https://github.com/hjwdzh/Manifold
