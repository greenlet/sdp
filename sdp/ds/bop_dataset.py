from enum import Enum

import multiprocessing as mpr
import shutil
import threading as thr
import traceback
from pathlib import Path
from typing import Optional, Union, Generator, Callable, Any, Generic, TypeVar, Iterator

import cv2
import imgaug.augmenters as iaa
from imageio.v3 import imread
import numpy as np
import pandas as pd

from sdp.ds.batch_processor import BatchProcessor
from sdp.ds.bop_data import read_scene_camera, read_scene_gt, read_scene_gt_info, id_to_str
from sdp.utils.data import read_yaml, write_yaml

SplitsType = Union[int, list[int], float, list[float]]
IdsType = Union[int, list[int], np.ndarray, range]


def get_img_fpath(scene_path: Path, img_id_str: str) -> Path:
    return scene_path / 'rgb' / f'{img_id_str}.jpg'


def get_mask_visib_fpath(scene_path: Path, img_id_str: str, obj_ind: int) -> Path:
    return scene_path / 'mask_visib' / f'{img_id_str}_{id_to_str(obj_ind)}.png'


def split_range(n: int, splits: SplitsType) -> list[int]:
    res = [0]
    if type(splits) == int:
        assert splits == -1 or 0 < splits <= n
        if splits == -1:
            return [0, n]
        div, rem = divmod(n, splits)
        i_split, i = 0, 0
        while i_split < splits:
            off = div
            if rem > 0:
                off += 1
                rem -= 1
            i += off
            res.append(i)
            i_split += 1
        return res

    if type(splits) == list and len(splits) == 0:
        return [0, n]

    if type(splits) == float:
        splits = [splits]

    if type(splits) == list and type(splits[0]) == float:
        spl = []
        was_neg, total = False, 0
        for s in splits:
            if s < 0:
                spl.append(-1)
                was_neg = True
            else:
                x = int(n * s)
                spl.append(x)
                total += x
        if not was_neg and total < n:
            spl.append(n - total)
        splits = spl

    if type(splits) == list and type(splits[0]) == int:
        was_neg, total = False, 0
        for s in splits:
            assert type(s) == int
            if s == -1:
                assert not was_neg
                was_neg = True
            else:
                assert s > 0
                total += s
        assert was_neg and total < n or total == n, f'was_neg: {was_neg}. total: {total}. n: {n}'
        i, rest = 0, n - total
        for s in splits:
            i += (s if s > 0 else rest)
            res.append(i)
        return res

    raise Exception(f'Unknown splits format: {splits}')


class ImgsMasksCrop:
    imgs_crop: list[np.ndarray]
    masks_crop: list[np.ndarray]
    bboxes_ltwh_src: list[np.ndarray]
    ratios_src_to_dst: list[float]

    def __init__(self, imgs_crop: list[np.ndarray], masks_crop: list[np.ndarray],
                 bboxes_ltwh_src: list[np.ndarray], ratios_src_to_dst: list[float]):
        self.imgs_crop = imgs_crop
        self.masks_crop = masks_crop
        self.bboxes_ltwh_src = bboxes_ltwh_src
        self.ratios_src_to_dst = ratios_src_to_dst


def crop_apply_masks(imgs: list[np.ndarray], masks: list[np.ndarray], bboxes_ltwh: list[np.ndarray], patch_sz: int,
                     offset: float = 0.05) -> ImgsMasksCrop:
    n = len(imgs)
    imgs_crop = [np.zeros((patch_sz, patch_sz, 3), dtype=np.uint8) for _ in range(n)]
    masks_crop = [np.zeros((patch_sz, patch_sz), dtype=np.uint8) for _ in range(n)]
    bboxes_ltwh_src = [np.zeros(4, dtype=float) for _ in range(n)]
    ratios_src_to_dst = []
    r = 1 + offset
    for img, mask, bbox_ltwh, img_crop, mask_crop, bbox_ltwh_src in zip(imgs, masks, bboxes_ltwh, imgs_crop, masks_crop, bboxes_ltwh_src):
        img_sz = np.array((img.shape[1], img.shape[0]))
        bbox_center = bbox_ltwh[:2] + bbox_ltwh[2:] / 2
        bbox_sz = np.max(bbox_ltwh[2:])
        bbox_sz_ext = bbox_sz * r
        bbox_sz_ext2 = bbox_sz_ext / 2
        bbox_sz_ext = bbox_sz_ext.astype(int)
        bb_lt = (bbox_center - bbox_sz_ext2).astype(int)
        bb_rb = bb_lt + bbox_sz_ext - 1
        bb_lt_fit, bb_rb_fit = np.maximum(bb_lt, 0), np.minimum(bb_rb, img_sz - 1)
        bb_sz_fit = bb_rb_fit - bb_lt_fit + 1
        bb_lt_off = bb_lt_fit - bb_lt
        img_patch = np.zeros((bbox_sz_ext, bbox_sz_ext, 3), np.uint8)
        mask_patch = np.zeros((bbox_sz_ext, bbox_sz_ext), np.uint8)
        p_p1, p_p2 = bb_lt_off, bb_lt_off + bb_sz_fit
        i_p1, i_p2 = bb_lt_fit, bb_lt_fit + bb_sz_fit
        slices_patch = slice(p_p1[1], p_p2[1]), slice(p_p1[0], p_p2[0])
        slices_img = slice(i_p1[1], i_p2[1]), slice(i_p1[0], i_p2[0])
        img_patch[slices_patch] = img[slices_img]
        mask_patch[slices_patch] = mask[slices_img]
        cv2.resize(img_patch, img_crop.shape[:2], img_crop, interpolation=cv2.INTER_AREA)
        cv2.resize(mask_patch, mask_crop.shape, mask_crop, interpolation=cv2.INTER_NEAREST)
        bbox_ltwh_src[:2] = bb_lt_fit
        bbox_ltwh_src[2:] = bb_sz_fit
        ratios_src_to_dst.append(patch_sz / bbox_sz_ext)
    return ImgsMasksCrop(imgs_crop, masks_crop, bboxes_ltwh_src, ratios_src_to_dst)


class GtImgsMasks:
    df_img: pd.DataFrame
    df_obj: pd.DataFrame
    imgs: list[np.ndarray]
    masks: list[np.ndarray]
    imgs_crop: list[np.ndarray]
    masks_crop: list[np.ndarray]
    bboxes_ltwh_src: list[np.ndarray]
    ratios_src_to_dst: list[float]

    def __init__(self, df_img: pd.DataFrame, df_obj: pd.DataFrame,
                 imgs: list[np.ndarray], masks: list[np.ndarray]):
        self.df_img = df_img
        self.df_obj = df_obj
        self.imgs = imgs
        self.masks = masks
        self.imgs_crop = []
        self.masks_crop = []
        self.bboxes_ltwh_src = []
        self.ratios_src_to_dst = []

    def crop(self, patch_sz: int, offset: float = 0.05):
        img_masks_crop = crop_apply_masks(self.imgs, self.masks, self.df_obj['bbox_visib_ltwh'], patch_sz, offset)
        self.imgs_crop = img_masks_crop.imgs_crop
        self.masks_crop = img_masks_crop.masks_crop
        self.bboxes_ltwh_src = img_masks_crop.bboxes_ltwh_src
        self.ratios_src_to_dst = img_masks_crop.ratios_src_to_dst


def create_aug_default() -> iaa.Augmenter:
    aug = iaa.Sequential([
        iaa.OneOf([
            iaa.Add((-50, 100)),
            iaa.Add((-50, 100), per_channel=True),
            iaa.Multiply((0.8, 1.8)),
        ]),
        iaa.GaussianBlur(sigma=(0, 5)),
        iaa.LinearContrast((0.5, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0, 15)),
    ], random_order=True)
    return aug


AUGNAME_DEFAULT = 'default'
AUGS: dict[str, iaa.Augmenter] = {}


def get_aug(aug_name: str) -> iaa.Augmenter:
    if aug_name not in AUGS:
        if aug_name == AUGNAME_DEFAULT:
            AUGS[aug_name] = create_aug_default()
        else:
            raise Exception(f'Unknown aug_type = {aug_name}')

    return AUGS[aug_name]


class ImgsObjsGtParams:
    ds_path: Path
    irows: pd.DataFrame
    orows: pd.DataFrame
    out_size: Optional[int] = None
    aug_name: Optional[str] = None

    def __init__(self, ds_path: Path, irows: pd.DataFrame, orows: pd.DataFrame, out_size: Optional[int] = None,
                 aug_name: Optional[str] = None):
        self.ds_path = ds_path
        self.irows = irows.copy()
        self.orows = orows.copy()
        self.out_size = out_size
        self.aug_name = aug_name


def load_imgs_objs_gt(params: ImgsObjsGtParams) -> GtImgsMasks:
    imgs, masks = [], []
    for img_ds_id, irow in params.irows.iterrows():
        orows_img = params.orows[params.orows['img_ds_id'] == img_ds_id]
        scene_id_str = id_to_str(irow['scene_id'])
        img_id_str = id_to_str(irow['img_id'])
        scene_path = params.ds_path / scene_id_str
        img_fpath = get_img_fpath(scene_path, img_id_str)
        img = imread(img_fpath)
        imgs.append(img)
        mask_img = None
        for _, orow in orows_img.iterrows():
            if orow['px_count_visib'] == 0:
                continue
            mask_fpath = get_mask_visib_fpath(scene_path, img_id_str, orow['obj_ind'])
            mask = imread(mask_fpath)
            if mask_img is None:
                mask_img = mask
            mask_img[mask > 0] = orow['obj_ind'] + 1
        masks.append(mask_img)

    if params.aug_name is not None:
        aug = get_aug(params.aug_name)
        imgs = aug(images=imgs)

    res = GtImgsMasks(params.irows, params.orows, imgs, masks)
    if params.out_size is not None:
        res.crop(params.out_size)

    return res


class BopView:
    ds: 'BopDataset'
    ids: np.ndarray
    batch_size: Optional[int] = None
    aug_name: Optional[str] = None

    def __init__(self, ds: 'BopDataset', ids: np.ndarray, batch_size: Optional[int] = None, aug_name: Optional[str] = None):
        self.ds = ds
        self.ids = ids.copy()
        self.batch_size = batch_size
        self.aug_name = aug_name

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def shuffle(self):
        np.random.shuffle(self.ids)

    def __len__(self):
        return len(self.ids)

    def get_gt_task_params(self, i: IdsType, n: Optional[int] = None) -> ImgsObjsGtParams:
        raise Exception('Unimplemented')

    def get_gt_imgs_masks(self, i: IdsType, n: int) -> GtImgsMasks:
        params = self.get_gt_task_params(i, n)
        params.aug_name = self.aug_name
        return load_imgs_objs_gt(params)

    def get_gt_batch(self, i_batch: int) -> GtImgsMasks:
        assert self.batch_size is not None
        i = i_batch * self.batch_size
        return self.get_gt_imgs_masks(i, self.batch_size)

    def get_batch_iterator(self, n_batches: Optional[int] = None, multiprocess: bool = False, batch_size: Optional[int] = None,
                           drop_last: bool = False, shuffle_between_loops: bool = True, buffer_sz: int = 3,
                           aug_name: Optional[str] = None) -> Generator[GtImgsMasks, None, None]:
        batch_size = batch_size if batch_size is not None else self.batch_size
        aug_name = aug_name if aug_name is not None else self.aug_name
        n = len(self.ids)
        n_batches_total = n // batch_size + min(n % batch_size, 1)

        info = f'n = {n}. batch_size = {batch_size}. n_batches = {n_batches}. n_batches_total = {n_batches_total}'
        assert n_batches_total > 0, info
        assert n_batches is None or n_batches > 0, info

        looped = False
        if n_batches is None:
            n_batches = n_batches_total
        elif n_batches > n_batches_total:
            looped = True

        def batch_gen() -> Generator[ImgsObjsGtParams, None, None]:
            for i_batch in range(n_batches):
                i = i_batch * batch_size
                if i >= n:
                    if shuffle_between_loops:
                        np.random.shuffle(self.ids)
                        i = 0
                    else:
                        i %= n
                batch_size_cur = min(batch_size, n - i)
                inds = range(i, i + batch_size_cur)
                if batch_size_cur < batch_size:
                    if not looped:
                        if drop_last:
                            return
                    else:
                        rest = batch_size - batch_size_cur
                        inds = list(range(i, n)) + list(range(rest))
                params = self.get_gt_task_params(inds)
                params.aug_name = aug_name
                yield params

        batch_it = iter(batch_gen())
        if multiprocess:
            pool = self.ds.acquire_pool()
            bproc = BatchProcessor(load_imgs_objs_gt, pool, batch_it, buffer_sz=buffer_sz)
            for res in bproc:
                yield res
        else:
            for params in batch_it:
                res = load_imgs_objs_gt(params)
                yield res


class BopImgsView(BopView):
    def __init__(self, ds: 'BopDataset', ids: np.ndarray, batch_size: Optional[int] = None, aug_name: Optional[str] = None):
        super().__init__(ds, ids, batch_size, aug_name)

    def get_gt_task_params(self, i: IdsType, n: Optional[int] = None) -> ImgsObjsGtParams:
        if type(i) in (list, np.ndarray, range):
            assert n is None
            inds = i
        else:
            assert n is not None and n > 0
            inds = slice(i, i + n)
        img_ds_ids = self.ids[inds]
        dfi, dfo = self.ds.df_img, self.ds.df_obj
        irows = dfi.loc[img_ds_ids]
        orows = dfo[dfo['img_ds_id'].isin(img_ds_ids)]
        return ImgsObjsGtParams(self.ds.get_ds_path(), irows, orows)


class BopObjsView(BopView):
    obj_ids: np.ndarray
    out_size: Optional[int] = None
    min_bbox_dim_ratio: Optional[float] = None
    min_mask_ratio: Optional[float] = None

    def __init__(self, ds: 'BopDataset', ids: Optional[np.ndarray] = None, obj_ids: Optional[IdsType] = None,
                 batch_size: Optional[int] = None, out_size: Optional[int] = None, min_bbox_dim_ratio: Optional[float] = None,
                 min_mask_ratio: Optional[float] = None, aug_name: Optional[str] = None):
        if min_bbox_dim_ratio is not None: assert min_bbox_dim_ratio > 0
        if min_mask_ratio is not None: assert min_mask_ratio > 0

        if ids is not None:
            assert obj_ids is None, f'Either ids or obj_ids can have non-None value. Input provided: ' \
                                    f'ids = {ids}, obj_ids = {obj_ids}'
            df_obj = ds.df_obj.loc[ids]
            obj_ids = df_obj['obj_id'].unique()

        if obj_ids is not None:
            obj_ids_all = ds.df_obj['obj_id'].unique()
            if obj_ids == -1:
                obj_ids_common = obj_ids_all
            else:
                if type(obj_ids) == int:
                    obj_ids = [obj_ids]
                obj_ids_diff = np.setdiff1d(obj_ids, obj_ids_all)
                if len(obj_ids_diff) > 0:
                    print(f'Warning. Input obj_ids = {obj_ids}. Dataset obj_ids: {obj_ids_all}. There are input obj_ids '
                          f'which are not present in dataset: {obj_ids_diff}')
                obj_ids_common = np.intersect1d(obj_ids, obj_ids_all)
            assert len(obj_ids_common) > 0, f'Warning. Input obj_ids = {obj_ids}. Dataset obj_ids: {obj_ids_all}. ' \
                                            f'Their intersection is emtpy'
            inds = ds.df_obj['obj_id'].isin(obj_ids_common)
            ids = ds.df_obj[inds].index.values

        if min_mask_ratio is not None or min_bbox_dim_ratio is not None:
            dfo = ds.df_obj.loc[ids]
            dfo = dfo[dfo['px_count_visib'] > 0]
            dfi = ds.df_img.loc[dfo['img_ds_id']]
            if min_mask_ratio is not None:
                img_area = dfi['img_width'] * dfi['img_height']
                inds = dfo['px_count_visib'].values / img_area.values >= min_mask_ratio
                dfo, dfi = dfo[inds], dfi[inds]
            if min_bbox_dim_ratio is not None:
                bbox_ltwh = np.stack(dfo['bbox_visib_ltwh'])
                bbox_sz = bbox_ltwh[:, 2:].min(axis=1)
                img_area = dfi[['img_width', 'img_height']].max(axis=1)
                inds = bbox_sz / img_area.values >= min_bbox_dim_ratio
                dfo = dfo[inds]
            ids = dfo.index.values

        self.min_mask_ratio = min_mask_ratio
        self.min_bbox_dim_ratio = min_bbox_dim_ratio
        self.out_size = out_size
        super().__init__(ds, ids, batch_size, aug_name)
        self.obj_ids = obj_ids

    def set_out_size(self, out_size: int):
        self.out_size = out_size

    def get_gt_task_params(self, i: IdsType, n: Optional[int] = None) -> ImgsObjsGtParams:
        if type(i) in (list, np.ndarray, range):
            assert n is None
            inds = i
        else:
            assert n is not None and n > 0
            inds = slice(i, i + n)
        obj_ds_ids = self.ids[inds]
        orows = self.ds.df_obj.loc[obj_ds_ids]
        img_ds_ids = pd.unique(orows['img_ds_id'])
        irows = self.ds.df_img.loc[img_ds_ids]
        return ImgsObjsGtParams(self.ds.get_ds_path(), irows, orows, self.out_size)


class BopDataset:
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

    pool: Optional[mpr.Pool] = None
    pool_refcount = 0

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
            self.df_img[img_colnames].to_csv(ds_img_fpath)
            self.df_obj[obj_colnames].to_csv(ds_obj_fpath)
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
            df_img.set_index('img_ds_id', inplace=True)
            df_obj.set_index('obj_ds_id', inplace=True)
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
        img_keys = [
            'img_ds_id',
            'scene_id',
            'img_id',
            'cam_K',
            'cam_depth_scale',
            'img_width',
            'img_height',
        ]
        obj_data = []
        obj_keys = [
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
        ]
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

            first_img_fpath = get_img_fpath(scene_path, id_to_str(imgs_ids[0]))
            first_img = imread(first_img_fpath)
            img_width, img_height = first_img.shape[1], first_img.shape[0]

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
                    img_width,
                    img_height,
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
        df_img.set_index('img_ds_id', inplace=True)
        df_obj.set_index('obj_ds_id', inplace=True)

        ds = BopDataset(bop_path, ds_name, ds_subdir, df_img, df_obj, False)
        written_to_cache = False
        if shuffle:
            written_to_cache = ds.shuffle()
        if not written_to_cache:
            ds.write_cache()

        return ds

    def get_imgs_view(self, batch_size: Optional[int] = None, aug_name: Optional[str] = None) -> BopImgsView:
        return BopImgsView(self, self.df_img.index.values, batch_size, aug_name=aug_name)

    def get_objs_view(self, obj_ids: IdsType = -1, batch_size: Optional[int] = None,
                      out_size: Optional[int] = None, min_bbox_dim_ratio: Optional[float] = 0.05,
                      min_mask_ratio: Optional[float] = 0.001, aug_name: Optional[str] = None) -> BopObjsView:
        return BopObjsView(self, obj_ids=obj_ids, batch_size=batch_size, out_size=out_size, min_bbox_dim_ratio=min_bbox_dim_ratio,
                           min_mask_ratio=min_mask_ratio, aug_name=aug_name)

    def get_ds_path(self) -> Path:
        return self.bop_path / self.ds_name / self.ds_subdir

    def get_scene_path(self, scene_id: int) -> Path:
        return self.bop_path / self.ds_name / self.ds_subdir / id_to_str(scene_id)

    def acquire_pool(self) -> mpr.Pool:
        if self.pool_refcount == 0:
            self.pool = mpr.Pool()
        self.pool_refcount += 1
        return self.pool

    def release_pool(self):
        if self.pool_refcount > 0:
            self.pool_refcount -= 1
            if self.pool_refcount == 0:
                self.pool.terminate()
                self.pool = None

