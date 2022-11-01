'''
author:
    zhangsihao yang
logs:
    20220926
        file created
'''
import sys

sys.path.insert(0, '/workspace')
import numpy as np
import torch
from lib.datasets.modelnet40.meshnet2.ssl_rec_att import Dataset
from tqdm import tqdm

dataset = Dataset(partition='test')

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, num_workers=0, 
    collate_fn=dataset.__collate__,
    drop_last=False
)

from lib.models.modelnet40.meshnet2.ssl_mae import Network

network = Network(
    en_config={
        'num_faces': 1024,
        'num_cls': 40,
        'cfg': {
            'num_kernel': 64,
            'ConvSurface': {
                'num_samples_per_neighbor': 4,
                'rs_mode': 'Weighted',
                'num_kernel': 64
            },
            'MeshBlock': {
                'blocks': [3, 4, 4]
            }
        },
        'pool_rate': 4,
        'mask_percentage': 0.0,
    },
    de_config={
        'bneck_size': 1024
    }
)

ckpt_path = '/runtime/modelnet40/meshnet2/ssl_mae/0_6/_290.pth'
ckpt = torch.load(ckpt_path)
net_state = ckpt['net']
# print(net_state)
from collections import OrderedDict

d2 = OrderedDict(
    [
        (k[len('module.'):], v) if k.startswith('module.') else (k, v) for k, v in net_state.items()
    ]
)

network.load_state_dict(d2)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(data_subset)
list_fea = []
list_target = []

counter = 0
for data in tqdm(dataloader):
    input_data, output_data = data
    
    y, fea = network(**input_data)

    target = output_data['targets'].detach().cpu().numpy()

    list_fea.append(fea[target < 10])
    list_target.append(target[target < 10])

    counter += 1
    if counter > 20:
        break

# concat all the results together
cat_fea = torch.cat(list_fea).detach().numpy()
cat_label = np.concatenate(list_target)

tsne_results = tsne.fit_transform(cat_fea)
# tsne_results = np.array(tsne_results)
# plt.figure(figsize=(16,10))
# df = {'x': tsne_results, 'y': target}
import pandas as pd

df = pd.DataFrame()
df["comp-1"] = tsne_results[:, 0]
# print()
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]
df['comp-2'] = tsne_results[:, 1]
import matplotlib
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
import seaborn as sns

sns.set()
# df = sns.load_dataset('iris')
sns_plot = sns.scatterplot(
    x="comp-1", y="comp-2",
    hue=cat_label.tolist(),
    palette=sns.color_palette("hls", 10),
    data=df,
    legend=False,
    # alpha=0.3
)
plt.legend([],[], frameon=False)
fig = sns_plot.get_figure()
fig.savefig("output.png")
#sns.plt.show()


# print(fea.shape)
