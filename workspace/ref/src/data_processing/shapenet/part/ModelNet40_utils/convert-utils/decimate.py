"""
Author: Vinit V. Singh
Wrapper around https://github.com/hjwdzh/Manifold to decimate waterlight mesh
to desired number of faces
"""

import os
import subprocess

#[1]
decimate = '/media/jakep/Elements/Mesh/MeshNet++/Decimate/Manifold/build/simplify'

#Path to waterlight mesh
src = '../ModelNet40-waterlight/'
dst = '../ModelNet40-decimate-1024/'

#Desired number of faces
max_faces = 1024

for root, dirs, files in os.walk(src, topdown = False):
    for name in files:
        if name.endswith('.obj'):
            o = root.replace(src, dst)
            if not os.path.exists(o):
                os.makedirs(o)
            i = root + os.sep + name
            o = o + os.sep + name
            cmd = decimate + ' -i ' + i + ' -o ' + o + ' -m ' + ' -f ' + str(max_faces)
            subprocess.call(cmd, shell=True)

#References:
#1. https://github.com/hjwdzh/Manifold
