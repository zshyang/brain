'''
author:
    zhangsihao yang

date:
    20220823

logs:
    20220823
        file    created
'''


def under_development():
    ''' the purpose of this function is 
    make merge sparse feature with different 
    shapes but from the same initial input.
    '''
    def merge_two_sparse_tensor(tensor0, tensor1):
        assert tensor0.dense().shape == tensor1.dense().shape
        indices = tensor0.indices.long()
        dense1 = tensor1.dense()
        feature1 = dense1[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]
        tensor0.features = tensor0.features + feature1
        return tensor0

    def get_grid_coords(sp_input):
        d0 = torch.linspace(-1.0, 1.0, sp_input['spshape'][0])
        d1 = torch.linspace(-1.0, 1.0, sp_input['spshape'][1])
        d2 = torch.linspace(-1.0, 1.0, sp_input['spshape'][2])

        meshx, meshy, meshz = torch.meshgrid((d0, d1, d2))
        grid = torch.stack((meshy, meshx, meshz), -1)

        grid = grid.unsqueeze(0) # add batch dim
        grid = grid.repeat((sp_input['batch_size'], 1, 1, 1, 1))

        grid = grid.to(sp_input['features'].device)

        return grid

    def interpolate_features(grid_coords, feature_volume):
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(
                volume, grid_coords,
                padding_mode='zeros', 
                align_corners=True
            )
            features.append(feature)
        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4))
        return features

    def test(grid, sparse_tensor, sp_input):
        pass
