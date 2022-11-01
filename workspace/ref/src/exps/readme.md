folder structure
```
.
|-- jigsaw/
|   |-- moco/
|   |   |-- pointnet/
|   |   |   |-- data_augmentation/
|   |   |   |-- loss/
|   |   |   `-- structure/
|   |   |       |-- 0.yaml
|   |   |       |-- 1.yaml
|   |   |       |-- 2.yaml
|   |   |       |-- 3.yaml
|   |   |       `-- debug.yaml
|   |   |-- pcn_debug.yaml
|   |   |-- pcn_test.yaml
|   |   |-- pcn.yaml
|   |   |-- pointnet_debug.yaml
|   |   |-- pointnet_test.yaml
|   |   `-- pointnet.yaml
|   |-- debug.yaml
|   |-- pointnet_test.yaml
|   `-- pointnet.yaml
|-- mesh/
|-- occo/
|   |-- finetune/
|   |   `-- pn_mn.yaml
|   `-- pretrain/
|       `-- p.yaml : pretrain
|-- pnvae_shapenet/ : pretrain pointnet vae on shapenet
|-- debug.yaml
|-- min.yaml
|-- pnvae_shapenet.yaml
`-- pointnet_modelnet.yaml : reproduce PointNet on ModelNet40
```

0 stands for 00
1 stands for 01
2 stands for 10
3 stands for 11
