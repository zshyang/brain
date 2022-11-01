import os

from lib.vis.htmlviewer.scenes_generator import ScenesGenerator


def view_pc(root, pc0, pc1, pc_path):
    root = '/home/george/George/projects/base/runtime/multi-view/debug/data'
    # save point clouds
    rp0 = save_pc(pc_path, pc0, '0')
    rp1 = save_pc(pc_path, pc1, '1')

    # generate information
    lid = glid(rp0, rp1)

    # generate html
    otf = '/runtime/multi-view/debug/html'
    nnp = 5
    sg = ScenesGenerator(
        root, lid, otf, nnp
    )
    pass


def glid(rp0, rp1):
    lid = []
    for _rp0, _rp1 in zip(rp0, rp1):
        _lid = []
        _rp0d = {
            'desc': 'occo',
            'caption': 'pc',
            'url': _rp0,
            'type': 'pointcloud',
        }
        _rp1d = {
            'desc': 'jigsaw',
            'caption': 'pc',
            'url': _rp1,
            'type': 'pointcloud',
        }
        _lid.append(_rp0d)
        _lid.append(_rp1d)
        lid.append(_lid)
    return lid


def save_pc(save_path, pc, extra):
    return_path = []
    for i in range(len(pc)):
        _pc = pc[i]
        _save_path = os.path.join(
            save_path, extra,
            f'{i}.xyz'
        )
        os.makedirs(
            os.path.dirname(
                _save_path
            ),
            exist_ok=True
        )
        with open(
            _save_path, 'w'
        ) as of:
            for _p  in _pc:
                of.write(
                    f'{_p[0]} \t'
                    f'{_p[1]} \t'
                    f'{_p[2]} \t'
                    '\n'
                )
        return_path.append(
            os.path.relpath(
                _save_path,
                save_path
            )
        )

    return return_path
