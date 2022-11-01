this folder should have same structure as exps
```
.
|-- jigsaw/
|   |-- dgcnn/
|   |-- moco/
|   |   |-- dgcnn/
|   |   |-- pcn/
|   |   |   |-- a.sh
|   |   |   `-- s.sh
|   |   |-- pcn_debug/
|   |   |   `-- d.sh
|   |   |-- pcn_test/
|   |   |   `-- d.sh
|   |   |-- pointnet/
|   |   |   |-- a.sh
|   |   |   `-- s.sh
|   |   |-- pointnet_debug/
|   |   |   `-- d.sh
|   |   `-- pointnet_test/
|   |       `-- d.sh
|   |-- pointnet/
|   |   |-- a.sh
|   |   |-- d.sh
|   |   |-- l.sh
|   |   `-- s.sh
|   |-- pointnet_debug/
|   |   `-- d.sh
|   `-- pointnet_test/
|       `-- d.sh
|-- occo
|   |-- finetune
|   |   `-- pn_mn
|   |       |-- d.sh : debug in local 
|   |       |       docker
|   |       |-- s.sh : script runs in 
|   |       |       singularity image
|   |       |-- l.sh : script to 
|   |       |       launch sing.sh
|   |       `-- a.sh : script to run 
|   |               on Agave
|   `-- pretrain
|       |-- d.sh : debug
|       `-- pretrain.sh
|-- sbatch
`-- sh
```
