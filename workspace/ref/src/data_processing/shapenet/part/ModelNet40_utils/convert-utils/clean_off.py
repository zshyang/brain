"""
Author: Vinit V. Singh
Clean OFF files in ModelNet40
"""

import os
import subprocess

src = '../ModelNet40-model/'

for root, dirs, files in os.walk(src, topdown = False):
    for name in files:
        if name.endswith('.off'):
            pth_off_file = os.path.join(root, name)
            off_file = open(pth_off_file, 'r')
            lines = off_file.readlines()
            first_line = lines[0].strip('\n')
            if first_line != 'OFF':
                lines.insert(1, first_line.replace('OFF', '') + '\n')
                lines[0] = 'OFF' + '\n'
                off_file = open(pth_off_file, "w")
                for line in lines:
                    off_file.write(line)
                off_file.close()
