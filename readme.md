# Pre-Training Graph Attention Convolution for Brain Structural Imaging Biomarker Analysis and Its Application to Alzheimer's Disease Pathology Identification


**Pre-Training Graph Attention Convolution for Brain Structural Imaging Biomarker Analysis and Its Application to Alzheimer's Disease Pathology Identification**


[Zhangsihao Yang](https://scholar.google.com/citations?user=VaRp0cMAAAAJ&hl=en), Yi Su, [Mohammad Farazi](https://scholar.google.com/citations?hl=en&user=lHa5pY4AAAAJ), Wenhui Zhu, Yanxi Chen, [Eric M Reiman](https://scholar.google.com/citations?user=I-Khl7AAAAAJ), Richard J Caselli, [Kewei Chen](https://scholar.google.com/citations?user=d83ZIzEAAAAJ&hl=zh-CN), [Yalin Wang](https://gsl.lab.asu.edu/), [Natasha Lepore](https://keck.usc.edu/faculty-search/natasha-lepore/)

**[Paper](paper/ISBI2023_UnsupervisedLearning.pdf)**


process the m files into obj file 

a docker file for MGMA is needed to copy to here.

the next step is to process obj files into npy file for MGMA.

The try to train the network. 

generate the docker image with the following command line:
```bash
docker-compose build brain
```

Next step is to generate data with obj format and visuliza some of them.

make the image up online
```bash
docker-compose up -d brain
```
enter the container
```bash
docker exec -it brain bash
```

debug inside of a docker container
```
python -m debugpy --listen 0.0.0.0:5566 --wait-for-client generate_obj.py
```

make the generate folder belong to my group
```bash
george@La:~/Projects/brain/workspace$ sudo chown -R george:george obj
```

There is a problem with the mesh. the mesh is not watertight.
![over all figure](fig/whole.png)
![detail figure](fig/crack.png)


copy the manifold to excute the manifold and simplification.

There are totally 841 mesh files.

Next step is to make the simplified mesh file into npy files.

still need to remove redundent lines in each file.


in folder `train`, by running train.sh and test.sh we could get the un-supervised learning results.
