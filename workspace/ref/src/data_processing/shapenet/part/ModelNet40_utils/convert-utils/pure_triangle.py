"""
Author: Vinit V. Singh
Wrapper around Meshlab Pure Triangle script to convert quad or other mesh to
triangular mesh
"""

import os
import subprocess

src = '../ModelNet40-model-obj/'
dst = '../ModelNet40-triangle/'
script = 'pure_triangle.mlx'

for root, dirs, files in os.walk(src, topdown = False):
    for name in files:
        if name.endswith('.obj'):
            o = root.replace(src, dst)
            if not os.path.exists(o):
                os.makedirs(o)
            i = root + os.sep + name
            o = o + os.sep + name
            cmd = 'meshlabserver' + ' -i ' + i + ' -o ' + o + ' -s ' + script
            subprocess.call(cmd, shell=True)
