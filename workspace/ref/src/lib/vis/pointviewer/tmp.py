import numpy as np
import trimesh

tm = trimesh.load(
    '/dataset/modelnet/off_to_obj/obj/airplane/train/airplane_0001/model.obj'
)
print(tm)
v = np.array(tm.vertices)
pick = np.random.choice(
    np.arange(v.shape[0]), 2048, False
)
v = v[pick]
print(v.min(0))
print(v.max(0))

with open('air_0001.xyz', 'w') as of:
    for vertex in v:
        of.write(
            f'{vertex[0]} \t'
            f'{vertex[1]} \t'
            f'{vertex[2]} \t'
            '\n'
        )

