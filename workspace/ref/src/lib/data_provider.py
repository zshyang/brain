''' data provider. wrap dataloader and 
dataset into one

author
    Zhangsihao Yang

date
    2022-0314

name convetion
    gi = gather index
    trs = train (data provider) state
    tss = test (data provider) state
    trdp = train data provider
    tsdp = test data provider
    vls = validation (data provider) state
    vldp = validation data provider
'''
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from lib.vis.moco_viewer import view_pc


class collateFunctions:
    @staticmethod
    def pointnet_modelnet_cdt(
        list_data
    ):
        point_sets = torch.stack([d[0] for d in list_data], 0)
        labels = torch.cat([d[1] for d in list_data], 0)

        qc = np.stack(
            [0 for x in list_data]
        ).astype(np.int32)

        qc = torch.from_numpy(qc).long()
        # kc = torch.from_numpy(kc).long()

        return {
            'x': point_sets,
            'qc': qc
        }, {'labels': labels}

    @staticmethod
    def xyz(list_data):
        coordinates_batch, features_batch, labels_batch = \
            ME.utils.sparse_collate(
                [d[0] for d in list_data],
                [d[0] for d in list_data],
                [d[1] for d in list_data],
                dtype=torch.float32)
        return {"coordinates": coordinates_batch,
                "features": features_batch,
                "labels": labels_batch}

    @staticmethod
    def solid(list_data):
        # TODO
        coordinates_batch, features_batch, labels_batch = \
            ME.utils.sparse_collate(
                [d[0] for d in list_data],
                [torch.ones_list(d[0]) for d in list_data],
                [d[1] for d in list_data],
                dtype=torch.float32)
        return {"coordinates": coordinates_batch,
                "features": features_batch,
                "labels": labels_batch}

    @staticmethod
    def pointnet_modelnet(list_data):
        point_sets = torch.stack([d[0] for d in list_data], 0)
        labels = torch.cat([d[1] for d in list_data], 0)
        return {'x': point_sets}, {'labels': labels}

    @staticmethod
    def modelnet_jigsaw(list_data):
        x = torch.stack(
            [d[0] for d in list_data],
            0
        )
        y = torch.cat(
            [d[1] for d in list_data], 
            0
        ).long()
        return {'x': x}, {'y': y}

    @staticmethod
    def pnvae_shapenet(list_data):
        '''
        x is the input augmented point cloud
        y is the target point cloud
        '''
        x = torch.stack([d[0] for d in list_data], 0)
        y = torch.stack([d[2] for d in list_data], 0)

        return {'x': x}, {'y': y}

    @staticmethod
    def occo_modelnet(list_data):
        # ids = np.stack([x[0] for x in list_data])
        inputs = np.stack(
            [x[1] for x in list_data]
        ).astype(np.float32)
        inputs = np.swapaxes(inputs, 1, 2)
        inputs = torch.from_numpy(inputs)
        # npts = np.stack(
        #     [x[1].shape[0] for x in list_data]
        # ).astype(np.int32)
        gts = np.stack(
            [x[2] for x in list_data]
        ).astype(np.float32)
        gts = torch.from_numpy(gts)
        # return ids, inputs, npts, gts
        return {'x': inputs}, {'y': gts}
    
    @staticmethod
    def occo_moco(list_data):
        q = np.stack(
            [x[0] for x in list_data]
        ).astype(np.float32)
        k = np.stack(
            [x[1] for x in list_data]
        ).astype(np.float32)

        view_pc(
            '', q, k,
            '/runtime/multi-view/debug/data'
        )

        q = np.swapaxes(q, 1, 2)
        k = np.swapaxes(k, 1, 2)

        q = torch.from_numpy(q)
        k = torch.from_numpy(k)

        return {'im_q': q, 'im_k': k}, {}
    
    @staticmethod
    def occo_jigsaw(list_data):
        '''
        name convention:
            qc = query condition
            day = data augmentation y
        '''
        q = np.stack(
            [x[0][0] for x in list_data]
        ).astype(np.float32)
        k = np.stack(
            [x[1][0] for x in list_data]
        ).astype(np.float32)

        ''' visualization code
        view_pc(
            '', q, k,
            '/runtime/multi-view/debug/data'
        )
        '''

        q = np.swapaxes(q, 1, 2)
        k = np.swapaxes(k, 1, 2)

        q = torch.from_numpy(q)
        k = torch.from_numpy(k)

        qc = np.stack(
            [x[0][1] for x in list_data]
        ).astype(np.int32)
        kc = np.stack(
            [x[1][1] for x in list_data]
        ).astype(np.int32)

        qc = torch.from_numpy(qc).long()
        kc = torch.from_numpy(kc).long()

        return {
            'im_q': q,
            'im_k': k,
            'qc': qc,
            'kc': kc,
        }, {'day': qc}

    @staticmethod
    def meshes_pc(list_data):
        vs = []
        vms = []
        fs = []
        fms = []
        gather_index = []
        pcs = []
        i = 1
        for meshes, pc, nb in list_data:
            pcs.append(torch.from_numpy(pc))
            idxs = np.zeros(
                (nb * nb * nb)
            )
            for idx, tm in meshes.items():
                # Fetch.
                v, f = tm
                if v.shape[0] == 0:
                    continue
                vm = np.ones_like(
                    v[..., 0]
                )
                fm = np.ones_like(f)
                # Append.
                vs.append(torch.from_numpy(v))
                vms.append(torch.from_numpy(vm))
                fs.append(torch.from_numpy(f))
                fms.append(torch.from_numpy(fm))
                # idxs.append(idx)
                idxs[idx] = i
                i = i + 1
            gather_index.append(idxs)
        # Pad.
        vs = pad_sequence(vs, batch_first=True)
        vms = 1- pad_sequence(vms, batch_first=True)
        fs = pad_sequence(fs, batch_first=True)
        fms = 1 - pad_sequence(fms, batch_first=True)
        gi = torch.from_numpy(
            np.stack(gather_index)
        ).type(torch.int64)
        y = torch.stack(pcs, 0)

        return {
            'vs': vs, 'vms': vms, 'fs': fs, 'fms': fms,
            'gi': gi
        }, {'y': y}


class DataProvider():
    '''
    Assume the data augmentation is done by numpy. When 
    storing the state, numpy random state should be stored 
    as well.
    '''
    def __init__(self, dataset, data_opt):
        collate_fn = getattr(
            collateFunctions, data_opt.collate_fn)
        sampler = ResumableRandomSampler(dataset)

        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=data_opt.batch_size,
            num_workers=int(data_opt.num_workers),
            collate_fn=collate_fn,
            drop_last=data_opt.drop_last, sampler=sampler)
        self.train_iter = iter(self.dataloader)

        self.dataset = dataset
        self.data_opt = data_opt

        self.collate_fn = collate_fn
        self.sampler = sampler

    def _create_input_batch(self, batch, device="cuda"):
        voxel_size = self.data_opt.voxel_size
        batch["coordinates"][:, 1:] = \
            batch["coordinates"][:, 1:] / voxel_size

        input_data = ME.TensorField(
            coordinates=batch["coordinates"],
            features=batch["features"],
            device=device)
        target = batch['labels'].to(device)

        return input_data, target

    def get_next_batch(self):
        try:
            data_dict = self.train_iter.next()
        except StopIteration:
            self.train_iter = iter(self.dataloader)
            data_dict = self.train_iter.next()

        input_batch = self._create_input_batch(data_dict)

        return input_batch

    def set_state(self, state_dict):
        np.random.set_state(state_dict['np'])

        sampler = ResumableRandomSampler(self.dataset)
        sampler.set_state(state_dict['sampler'])

        self.sampler = sampler

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.data_opt.batch_size,
            num_workers=int(self.data_opt.num_workers),
            collate_fn=self.collate_fn, 
            drop_last=self.data_opt.drop_last,
            sampler=sampler)
        self.train_iter = iter(self.dataloader)

    def get_state(self):
        '''
        Becuase self.dataloader.sampler is self.sampler. So
        we could get the state directly from self.sampler.
        '''
        sampler_state = self.sampler.get_state()
        np_state = np.random.get_state()
        return {'sampler': sampler_state, 'np': np_state}


class DistributedDataProvider():
    def __init__(self, opt):
        ''' 

        Args:
            opt
        '''

        self.opt = opt

        import lib.datasets as dataset
        dataset_lib = self.opt.dataset.lib
        self.dataset = getattr(
            dataset, dataset_lib
        )(
            **self.opt.dataset.params
        )

        self.collate_fn = getattr(
            collateFunctions, 
            str(self.opt.dataloader.collate_fn),
            None
        )
        if self.collate_fn is None:
            self.collate_fn = getattr(self.dataset, str(self.opt.dataloader.collate_fn), None)

        self.sampler = torch.utils.data.DistributedSampler(
            self.dataset, shuffle=True
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            batch_size=self.opt.dataloader.batch_size,
            drop_last=bool(self.opt.dataloader.drop_last),
            num_workers=int(self.opt.dataloader.num_workers)
        )
        self.train_iter = iter(self.dataloader)

        self.epoch = 0

    def __len__(self):
        return len(self.dataloader)

    def get_next_batch(self):
        try:
            input_batch = self.train_iter.next()
        except StopIteration:
            self.epoch = self.epoch + 1
            self.sampler.set_epoch(self.epoch)
            self.train_iter = iter(self.dataloader)
            input_batch = self.train_iter.next()

        return input_batch

    def set_state(self, state_dict):
        np.random.set_state(state_dict['np'])
        self.epoch = state_dict['epoch'] + 1
        self.sampler.set_epoch(self.epoch)
        self.train_iter = iter(self.dataloader)

    def get_state(self):
        '''
        Becuase self.dataloader.sampler is self.sampler. So
        we could get the state directly from self.sampler.
        '''
        np_state = np.random.get_state()
        return {'epoch': self.epoch, 'np': np_state}


class TVTDistributedDataProvider():
    def __init__(self, opt):
        '''
        Args:
            opt: from options.data
        '''
        if opt.train is None:
            self.trdp = None
        else:
            self.trdp = DistributedDataProvider(opt.train)

        if opt.val is None:
            self.vldp = None
        else:
            self.vldp = DistributedDataProvider(opt.val)

        if opt.test is None:
            self.tsdp = None
        else:
            self.tsdp = DistributedDataProvider(opt.test)
    
    def get_state(self):
        trs = self.trdp.get_state()

        if self.vldp is not None:
            vls = self.vldp.get_state()
        else:
            vls = None

        if self.tsdp is not None:
            tss = self.tsdp.get_state()
        else:
            tss = None

        return {'trs': trs, 'vls': vls, 'tss': tss}

    def set_state(self, state_dict):
        self.trdp.set_state(state_dict['trs'])

        if self.vldp is not None:
            self.vldp.set_state(
                state_dict['vls']
            )

        if self.tsdp is not None:
            self.tsdp.set_state(
                state_dict['tss']
            )
