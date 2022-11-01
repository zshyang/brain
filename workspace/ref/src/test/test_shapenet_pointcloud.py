import os
import sys

sys.path.append(os.path.abspath('.'))

from lib.dataset import ShapeNetPointCloud

shape_net = ShapeNetPointCloud(
    'train', None, None, 1, None)

for data in shape_net:
    print(data.shape)
    # pass

''' conclusion:
all the point cloud in shapenet point cloud dataset have 
2048 points.
'''