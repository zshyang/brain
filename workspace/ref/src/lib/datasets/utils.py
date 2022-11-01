import msgpack
import msgpack_numpy
import numpy as np
from lib.external.python_plyfile.plyfile import PlyData

msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000


def load_ply(file_name):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']

    points = np.vstack(
        [points['x'], points['y'], points['z']]
    ).T

    return points


def loads(buf):
    rb = msgpack.loads(
        buf, raw=False, 
        max_bin_len=MAX_MSGPACK_LEN,
        max_array_len=MAX_MSGPACK_LEN,
        max_map_len=MAX_MSGPACK_LEN,
        max_str_len=MAX_MSGPACK_LEN
    )
    return rb
