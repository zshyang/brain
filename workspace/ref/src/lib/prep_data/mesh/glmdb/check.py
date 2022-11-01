'''
interactive
module load singularity/3.8.0
singularity exec \
-B /scratch/zyang195/projects/base/src/:/workspace/ \
-B /scratch/zyang195/dataset/:/dataset/ \
/scratch/zyang195/singularity/occo_v2.simg \
bash


with len = 52472
target length = 51738
first round : 50972
'''
import json
from glob import glob

list_path = glob('/dataset/shapenet/voxsim/*/*/*.json')
print(len(list_path))
