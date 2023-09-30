import os
from typing import Union

import multiprocessing as mpr
from pathlib import Path

from imageio import imsave
import numpy as np
import pandas as pd

from sdp.ds.bop_data import read_meshes, id_to_str
from sdp.ds.bop_dataset import BopDataset
from sdp.lib3d.renderer import Renderer, OutputType


class NocNormsParams:
    worker_id: int
    ds_root_path: Path
    models_subdir: str
    img_ds_ids: Union[np.ndarray, list[int]]

    def __init__(self, worker_id: int, ds_root_path: Path, models_subdir: str, img_ds_ids: Union[np.ndarray, list[int]]):
        self.worker_id = worker_id
        self.ds_root_path = ds_root_path
        self.models_subdir = models_subdir
        self.img_ds_ids = img_ds_ids


def gen_noc_norms(params: NocNormsParams):
    print(f'Worker {params.worker_id} starting')
    cache_path = params.ds_root_path / BopDataset.cache_subdir
    ds = BopDataset.read_cache(cache_path)
    models_info_path = params.ds_root_path / params.models_subdir
    meshes = read_meshes(models_info_path)
    irows = ds.df_img.loc[params.img_ds_ids]
    orows = ds.df_obj
    irow = irows.iloc[0]
    img_sz = irow['img_width'], irow['img_height']
    ren = Renderer(meshes, img_sz, hide_window=True)
    scene_path_pre = None
    for img_ds_id, irow in irows.iterrows():
        orows_img = orows[orows['img_ds_id'] == img_ds_id]
        img_id_str = id_to_str(irow['img_id'])
        scene_path = ds.get_scene_path(irow['scene_id'])
        norm_path, noc_path = scene_path / 'rgb_norm', scene_path / 'rgb_noc'
        if scene_path != scene_path_pre:
            norm_path.mkdir(exist_ok=True, parents=True)
            noc_path.mkdir(exist_ok=True, parents=True)
            scene_path_pre = scene_path
        fname = f'{img_id_str}.png'
        norm_fpath, noc_fpath = norm_path / fname, noc_path / fname
        cam_mat = irow['cam_K']
        ren.set_intrinsics(cam_mat=cam_mat)
        mesh_poses = []
        for _, orow in orows_img.iterrows():
            if orow['px_count_visib'] == 0:
                continue
            obj_id = orow['obj_id']
            H = np.eye(4)
            H[:3, :3] = orow['R_m2c']
            H[:3, 3] = orow['t_m2c'] * meshes[obj_id].mul_to_meters
            mesh_poses.append((obj_id, H))

        ren.set_intrinsics(img_size=(irow['img_width'], irow['img_height']))
        img_norm = ren.gen_colors(cam_mat, mesh_poses, OutputType.Normals)
        img_noc = ren.gen_colors(cam_mat, mesh_poses, OutputType.Noc)
        # print(img_norm.shape, img_norm.min(), img_norm.mean(), img_norm.max())
        imsave(norm_fpath, img_norm)
        imsave(noc_fpath, img_noc)
    print(f'Worker {params.worker_id} stopping')


def gen_noc_params_mp(bop_path: Path, ds_name: str, n_proc: int = -1):
    n_proc = n_proc if n_proc > 0 else mpr.cpu_count()
    print(f'Number of processes: {n_proc}')
    ds_root_path = bop_path / ds_name
    cache_path = ds_root_path / BopDataset.cache_subdir
    ds = BopDataset.read_cache(cache_path)
    n_imgs = len(ds.df_img)
    intervals = np.linspace(0, n_imgs, n_proc + 1, dtype=int)
    ps = []
    for i in range(n_proc):
        rng = range(intervals[i], intervals[i + 1])
        ids = ds.df_img.index[rng]
        params = NocNormsParams(1, ds_root_path, 'models', ids)
        print(f'Starting process #{i} for {len(ids)} image ids')
        p = mpr.Process(target=gen_noc_norms, args=(params,))
        p.start()
        ps.append(p)

    print(f'Waiting for {len(ps)} processes to stop')
    for p in ps:
        p.join()
    print('Main process exiting')


def test_gen_noc_params():
    bop_path = Path(os.path.expandvars('$HOME/data/bop'))
    ds_root_path = bop_path / 'itodd'
    params = NocNormsParams(1, ds_root_path, 'models', np.arange(30))
    gen_noc_norms(params)


def test_gen_noc_params_mp():
    bop_path = Path(os.path.expandvars('$HOME/data/bop'))
    ds_name = 'itodd'
    gen_noc_params_mp(bop_path, ds_name)


if __name__ == '__main__':
    # test_gen_noc_params()
    test_gen_noc_params_mp()

