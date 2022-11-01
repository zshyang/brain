'''
interactive
module load singularity/3.8.0
singularity exec \
-B /scratch/zyang195/projects/base/src/:/workspace/ \
-B /scratch/zyang195/dataset/:/dataset/ \
/scratch/zyang195/singularity/large-mesh.sif \
bash

with len = 52472
'''
import json
from glob import glob

list_path = glob('/dataset/shapenet/ShapeNetCore.v2/*/*/models/model_normalized.obj')
print(len(list_path))

with open('/dataset/shapenet/mansim/imn.json', 'w') as fl:
    json.dump(list_path, fl)
