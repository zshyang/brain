'''
Zhangsihao Yang
04/16/2022

ds = dataset
op = output path
td = train dataset
vd = validation dataset
wf = write frequency
'''
import pickle

import lmdb
from lib.dataset.mesh_dataset import ShapeNet
from tqdm import tqdm
from torch.utils.data import DataLoader


def dumps(obj):
    return pickle.dumps(obj, protocol=-1)


def loads(buf):
    return pickle.loads(buf)


def dataset2lmdb(ds, op, wf=1000):
    # dataset = MeshDataset(list_path)

    data_loader = DataLoader(
        ds, num_workers=0, 
        collate_fn=lambda x: x,
    )

    # print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(
        op, subdir=False,
        map_size=1099511627776 * 2, 
        readonly=False,
        meminit=False, map_async=True
    )

    txn = db.begin(write=True)
    counter = 0
    for idx, data in tqdm(enumerate(data_loader)):
        # path, v, f = data[0]
        # if path:
        txn.put(
            u'{}'.format(idx).encode('ascii'), 
            dumps(data[0])
        )
        counter += 1
        if idx % wf == 0:
            print(f'[{idx}/{len(data_loader)}]')
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [
        u'{}'.format(k).encode('ascii') for k in range(
            counter + 1
        )
    ]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def main():
    # dp = '/scratch/zyang195/dataset/shapenet/mansim/'
    # generate_json(dp)
    # list_path = load_json(dp)
    # list_train, list_val = split_train_val(list_path)

    op = (
        '/dataset/shapenet/mansim/'
        'lmdb/train_64_2.lmdb'
    )
    td = ShapeNet(
        '/dataset/shapenet/',
        'train', 64, 2
    )
    dataset2lmdb(td, op, 1000)

    vd = ShapeNet(
        '/dataset/shapenet/',
        'val', 64, 2
    )
    op = (
        '/dataset/shapenet/mansim/'
        'lmdb/val_64_2.lmdb'
    )
    dataset2lmdb(vd, op, 1000)


if __name__ == '__main__':
    main()
