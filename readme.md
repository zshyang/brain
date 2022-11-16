## Pre-Training Graph Attention Convolution for Brain Structural Imaging Biomarker Analysis and Its Application to Alzheimer's Disease Pathology Identification

**Pre-Training Graph Attention Convolution for Brain Structural Imaging Biomarker Analysis and Its Application to Alzheimer's Disease Pathology Identification**

[Zhangsihao Yang](https://scholar.google.com/citations?user=VaRp0cMAAAAJ&hl=en), Yi Su, [Mohammad Farazi](https://scholar.google.com/citations?hl=en&user=lHa5pY4AAAAJ), Wenhui Zhu, Yanxi Chen, [Eric M Reiman](https://scholar.google.com/citations?user=I-Khl7AAAAAJ), Richard J Caselli, [Kewei Chen](https://scholar.google.com/citations?user=d83ZIzEAAAAJ&hl=zh-CN), [Yalin Wang](https://gsl.lab.asu.edu/), [Natasha Lepore](https://keck.usc.edu/faculty-search/natasha-lepore/)

**[Paper](paper/ISBI2023_UnsupervisedLearning.pdf)**

## Environment

We use `docker` and `docker-compose` for the reproducibility of our work.

Build the docker images and create a docker container.

``` bash
cd docker
docker-compose build brain
docker-compose up -d brain
```

## Data Processing

Download the folder from [here](https://drive.google.com/file/d/1f0T4OKhxGtIb4hbZEhB4vqTiKxETkuNJ/view?usp=sharing). Unzip the data folder under `workspace`. You would see `.m` in the folder named `MMS`. There are totally 841 mesh files.

Download [here](https://drive.google.com/file/d/1MnTKpQbDuxqhQDcFG9j1qZQJSk0rzPMC/view?usp=sharing) and unzip. You should see the folder with name `data`.

### 1. Process the m files into obj file

There are 2 options:

#### (a) Download the proccessed files.

Download the folder from [here](https://drive.google.com/file/d/1TU-FT1ptlhSWCJmwCV8aHHyyiAIhoDXr/view?usp=sharing). Unzip the data folder under `workspace`. You would get a folder named `obj`.

#### (b) Process the files locally.

Enter the docker image and process the files. `$` means the command running on local machine. `#` means the command running in the docker container.

```
$ docker exec -it brain bash
# cd /workspace/data_processing
# python generate_obj.py
---- run the following line if you want to debug in VSCode ----
# python -m debugpy --listen 0.0.0.0:5566 --wait-for-client generate_obj.py
```

There is a problem with the mesh. the mesh is not watertight. Thus we need the next step.
Overall                               | Zoom in
:------------------------------------:|:-------------------------------------:
<img src="fig/whole.png" width="200"> | <img src="fig/crack.png" width="200">


#### 2. Make the files watertight
You could download the files from [here](https://drive.google.com/file/d/1nKc08OyCc1L9WV-qGysKfw4b7WK4jBXz/view?usp=sharing) or run the follow commmand.
```
# python manifold.py
```

#### 3. Simplify the watertight meshes
You could download the files from [here](https://drive.google.com/file/d/1yWqaXiGweTCXO5VdJgVGUukllg9L4JKU/view?usp=sharing) or run the follow commmand.
```
# python simplify.py
```

#### 4. Process obj files into npy file as graph
You could download the files from [here](https://drive.google.com/file/d/1yWqaXiGweTCXO5VdJgVGUukllg9L4JKU/view?usp=sharing) or run the follow commmand.
Notice that simplified meshes and npy files are in the same folder.
```
# python generate_npy.py
```

#### 5. Sample point clouds from obj files
You could download the files from [here](https://drive.google.com/file/d/1AZpfz1oZYVJCn3xYqx2LzrNMUFbFGkpA/view?usp=sharing) or run the follow commmand.
```
# python sample_point.py
```

## Training the network 

### Pre-train

```
# ./workspace/train/exps/train.sh
```
The ckpt will be generated in folder `runtime`

For test with SVM and fine-tuning, you could need pre-trained ckpt file. 
We have provided one [here](https://drive.google.com/file/d/1pI440GmCx8Pbdi4KdTc_0sAVmQPhbI_r/view?usp=sharing) and you could put the file in the folder `runtime`.

### Test with SVM

Modify the path of the ckpt in file `single_gpu_test.py` to change the loaded model.
```
# ./workspace/train/exps/test.sh
```

### Fine-tuning

Fine-tune the network for compare group 1.
```
# ../workspace/finetuning/train_0.sh
```
Fine-tune for group 2 and 3. Change the device name in the script if needed.
```
# ./workspace/finetuning/train_1_GPU_0.sh
# ./workspace/finetuning/train_1_GPU_1.sh
# ./workspace/finetuning/train_2_GPU_2.sh
# ./workspace/finetuning/train_2_GPU_3.sh
```

## Issues

### simplifiy and manifold software

Refre to this [repo](https://github.com/hjwdzh/Manifold). You might need to replace file `manifold` and `simplify` in folder `workspace/data_processing/manifold` with your own compiled files.

### change the group of generate data files\
make the generated folder belong to your group

```bash
$ sudo chown -R {id_user}:{id_user} obj
```

### Figure 4 in paper

To generate the figure 4 in paper, run the following command and load the meshes in MeshLab.
```
# cd workspace/data_processing
# python simplify_example.py
```

## Citation

Please consider cite the following paper.

```latex
@inproceedings{yang2021deep,
  title={Deep Learning on SDF for Classifying Brain Biomarkers},
  author={Yang, Zhangsihao and Wu, Jianfeng and Thompson, Paul M and Wang, Yalin},
  booktitle={2021 43rd Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={1051--1054},
  year={2021},
  organization={IEEE}
}
```
