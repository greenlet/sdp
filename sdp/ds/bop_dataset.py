import numpy as np
import pandas as pd
import shutil
import traceback
from pathlib import Path
from typing import Optional

from sdp.ds.bop_data import read_scene_camera, read_scene_gt, read_scene_gt_info
from sdp.utils.data import read_yaml


class BopDataset:
    class View:
        ds: 'BopDataset'
        ids: pd.Series

        def __init__(self, ds: 'BopDataset', ids: pd.Series) -> None:
            self.ds = ds
            self.ids = ids
        
        def shuffle(self):
            self.ids = self.ids.sample(len(self))
        
        def __len__(self):
            return len(self.ids)

    class ImgsView(View):
        def get_item(i: int):
            pass
    
    class ObjsView(View):
        def get_item(i: int):
            pass

    version: str = 'v0.0.1'
    cache_subdir = f'.sdp/{version}'
    ds_img_cache_fname = 'img.csv'
    ds_obj_cache_fname = 'obj.csv'
    ds_info_cache_fname = 'info.yaml'
    ds_img_np_cols = {'cam_K': (3, 3)}
    ds_obj_np_cols = {'R_m2c': (3, 3), 't_m2c': (3,), 'bbox_obj_ltwh': (4,), 'bbox_visib_ltwh': (4,)}

    bop_path: Path
    ds_name: str
    ds_subdir: str
    df_img: pd.DataFrame
    df_obj: pd.DataFrame
    shuffled: bool

    def __init__(self, bop_path: Path, ds_name: str, ds_subdir: str, df_img: pd.DataFrame, df_obj: pd.DataFrame, shuffled: bool) -> None:
        self.bop_path = bop_path
        self.ds_name = ds_name
        self.ds_subdir = ds_subdir
        self.df_img = df_img
        self.df_obj = df_obj
        self.shuffled = shuffled
    
    @classmethod
    def get_cache_path(cls, bop_path: Path, ds_name: str) -> Path:
        return bop_path / ds_name / cls.cache_subdir

    # Returns if cache was written
    def shuffle(self) -> bool:
        if self.shuffled:
            return False
        self.df_img = self.df_img.sample(len(self.df_img))
        self.df_obj = self.df_obj.sample(len(self.df_obj))
        self.write_cache()
        return True

    @staticmethod
    def write_np_cols(df: pd.DataFrame, cols: dict[str, tuple[int, ...]], cache_path: Path, prefix: str) -> list[str]:
        rest_colnames = list(df.columns)
        for colname, _ in cols.items():
            fpath = cache_path / f'{prefix}_{colname}.npy'
            val = np.stack(df[colname].values)
            val.tofile(fpath)
            rest_colnames.remove(colname)
        return rest_colnames
    
    @staticmethod
    def read_np_cols(df: pd.DataFrame, cols: dict[str, tuple[int, ...]], cache_path: Path, prefix: str) -> pd.DataFrame:
        for colname, coldim in cols.items():
            fpath = cache_path / f'{prefix}_{colname}.npy'
            val = np.fromfile(fpath)
            val = val.reshape((-1,) + coldim)
            df[colname] = list(val)
        return df

    def write_cache(self):
        cache_path = self.get_cache_path(self.bop_path, self.ds_name)
        ds_img_fpath = cache_path / self.ds_img_cache_fname
        ds_obj_fpath = cache_path / self.ds_obj_cache_fname
        ds_info_fpath = cache_path / self.ds_info_cache_fname
        shutil.rmtree(cache_path, ignore_errors=True)
        print(f'Writing dataset to {cache_path}')
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            img_colnames = self.write_np_cols(self.df_img, self.ds_img_np_cols, cache_path, 'img')
            obj_colnames = self.write_np_cols(self.df_obj, self.ds_obj_np_cols, cache_path, 'obj')
            self.df_img[img_colnames].to_csv(ds_img_fpath, index=False)
            self.df_obj[obj_colnames].to_csv(ds_obj_fpath, index=False)
            info = {
                'bop_path': str(self.bop_path),
                'ds_name': self.ds_name,
                'ds_subdir': self.ds_subdir,
                'shuffled': self.shuffled,
            }
            write_yaml(info, ds_info_fpath)
        except Exception as e:
            traceback.print_exception(e)
    
    @classmethod
    def read_cache(cls, cache_path: Path) -> Optional['BopDataset']:
        print(f'Reading dataset from {cache_path}')
        ds_img_fpath = cache_path / cls.ds_img_cache_fname
        ds_obj_fpath = cache_path / cls.ds_obj_cache_fname
        ds_info_fpath = cache_path / cls.ds_info_cache_fname
        not_found = False
        if not ds_img_fpath.exists():
            not_found = True
            print(f'File {ds_img_fpath} does not exist')
        if not ds_obj_fpath.exists():
            not_found = True
            print(f'File {ds_obj_fpath} does not exist')
        if not ds_info_fpath.exists():
            not_found = True
            print(f'File {ds_info_fpath} does not exist')
        if not_found:
            return None
        try:
            df_img = pd.read_csv(ds_img_fpath)
            df_obj = pd.read_csv(ds_obj_fpath)
            df_img = cls.read_np_cols(df_img, cls.ds_img_np_cols, cache_path, 'img')
            df_obj = cls.read_np_cols(df_obj, cls.ds_obj_np_cols, cache_path, 'obj')
            info = read_yaml(ds_info_fpath)
            bop_path = Path(info['bop_path'])
            return BopDataset(bop_path, info['ds_name'], info['ds_subdir'], df_img, df_obj, info['shuffled'])
        except Exception as e:
            traceback.print_exception(e)
            return None

    @classmethod
    def from_dir(cls, bop_path: Path, ds_name: str, ds_subdir: str, shuffle: bool, skip_cache: bool = False) -> 'BopDataset':
        ds_path = bop_path / ds_name
        train_path = ds_path / ds_subdir
        cache_path = cls.get_cache_path(bop_path, ds_name)
        if skip_cache:
            print(f'Removing {cache_path}')
            shutil.rmtree(cache_path, ignore_errors=True)
        
        ds = None
        if cache_path.exists():
            ds = cls.read_cache(cache_path)
        
        if ds is not None:
            return ds

        img_data = []
        img_keys = (
            'img_ds_id',
            'scene_id',
            'img_id',
            'cam_K',
            'cam_depth_scale',
        )
        obj_data = []
        obj_keys = (
            'obj_ds_id',
            'img_ds_id',
            'obj_ind',
            'obj_id',
            'R_m2c',
            't_m2c',
            'bbox_obj_ltwh',
            'bbox_visib_ltwh',
            'px_count_all',
            'px_count_valid',
            'px_count_visib',
            'visib_fract',
        )
        scenes_paths = list(train_path.iterdir())
        scenes_paths.sort()
        img_ds_id, obj_ds_id = 0, 0
        for scene_path in scenes_paths:
            scene_id = int(scene_path.name)
            camera_fpath = scene_path / 'scene_camera.json'
            gt_fpath = scene_path / 'scene_gt.json'
            gt_info_fpath = scene_path / 'scene_gt_info.json'
            camera = read_scene_camera(camera_fpath)
            gt = read_scene_gt(gt_fpath)
            gt_info = read_scene_gt_info(gt_info_fpath)
            assert gt.keys() == gt_info.keys() == camera.keys()
            imgs_ids = list(gt.keys())
            imgs_ids.sort()
            for img_id in imgs_ids:
                cam_img = camera[img_id]
                gt_img = gt[img_id]
                gt_info_img = gt_info[img_id]
                assert len(gt_img) == len(gt_info_img)
                img_data.append((
                    img_ds_id,
                    scene_id,
                    img_id,
                    cam_img.K,
                    cam_img.depth_scale,
                ))
                for obj_ind in range(len(gt_img)):
                    gt_obj = gt_img[obj_ind]
                    gt_info_obj = gt_info_img[obj_ind]
                    obj_data.append((
                        obj_ds_id,
                        img_ds_id,
                        obj_ind,
                        gt_obj.obj_id,
                        gt_obj.R_m2c,
                        gt_obj.t_m2c,
                        gt_info_obj.bbox_obj,
                        gt_info_obj.bbox_visib,
                        gt_info_obj.px_count_all,
                        gt_info_obj.px_count_valid,
                        gt_info_obj.px_count_visib,
                        gt_info_obj.visib_fract,
                    ))
                    obj_ds_id += 1
                img_ds_id += 1
        df_img = pd.DataFrame(img_data, columns=img_keys)
        df_obj = pd.DataFrame(obj_data, columns=obj_keys)

        ds = BopDataset(bop_path, ds_name, ds_subdir, df_img, df_obj, False)
        written_to_cache = False
        if shuffle:
            written_to_cache = ds.shuffle()
        if not written_to_cache:
            ds.write_cache()

        return ds



