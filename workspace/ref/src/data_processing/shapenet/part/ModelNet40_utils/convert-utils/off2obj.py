"""
Author: Vinit V. Singh
Wrapper around off2obj function provided by Antiprism[1] Library
"""

import os
import subprocess

src = '../ModelNet40-model/'
dst = '../ModelNet40-model-obj/'

for root, dirs, files in os.walk(src, topdown = False):
    for name in files:
        if name.endswith('.off'):
            o = root.replace(src, dst)
            if not os.path.exists(o):
                os.makedirs(o)
            i = root + os.sep + name
            o = o + os.sep + name.replace('.off', '.obj')
            cmd = 'off2obj ' + i + ' > ' + o
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stdin=subprocess.PIPE,
                                       shell=True)
            process.communicate()

# References
#[1] https://www.antiprism.com/programs/off2obj.html
