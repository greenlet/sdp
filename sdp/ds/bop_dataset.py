from enum import Enum

import multiprocessing as mpr
import shutil
import threading as thr
import traceback
from pathlib import Path
from typing import Optional, Union, Generator, Callable, Any, Generic, TypeVar, Iterator

import cv2
import imgaug.augmenters as iaa
import torch
from imageio.v3 import imread
import numpy as np
import pandas as pd

from sdp.ds.batch_processor import BatchProcessor
from sdp.ds.bop_data import read_scene_camera, read_scene_gt, read_scene_gt_info, id_to_str
from sdp.utils.data import read_yaml, write_yaml

MAP_NORM_NAME = 'norm'
MAP_NOC_NAME = 'noc'
SplitsType = Union[int, list[int], float, list[float]]
IndsList = Union[list[int], np.ndarray, slice]
IdsType = Union[int, IndsList]
ImgsList = list[np.ndarray]
MapsDict = dict[str, ImgsList]


def get_img_fpath(scene_path: Path, img_id_str: str, dir_sfx: str = '', ext='jpg') -> Path:
    dir_name = 'rgb'
    if dir_sfx:
        dir_name = f'{dir_name}_{dir_sfx}'
    return scene_path / dir_name / f'{img_id_str}.{ext}'


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
    imgs_crop: ImgsList
    masks_crop: ImgsList
    maps_crop: MapsDict
    bboxes_ltwh_src: list[np.ndarray]
    ratios_src_to_dst: list[float]

    def __init__(self, imgs_crop: ImgsList, masks_crop: ImgsList, maps_crop: MapsDict,
                 bboxes_ltwh_src: list[np.ndarray], ratios_src_to_dst: list[float]):
        self.imgs_crop = imgs_crop
        self.masks_crop = masks_crop
        self.maps_crop = maps_crop
        self.bboxes_ltwh_src = bboxes_ltwh_src
        self.ratios_src_to_dst = ratios_src_to_dst


def make_square_img(sz: int, ch: int) -> np.ndarray:
    shape = (sz, sz) if ch == 1 else (sz, sz, ch)
    return np.zeros(shape, dtype=np.uint8)


def crop_apply_mask(imgs: list[np.ndarray], mask: np.ndarray, bbox_ltwh: np.ndarray, sz_out: int,
                    bb_ext_offset: float = 0.05) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
    imgs_out = [make_square_img(sz_out, 3) for _ in imgs]
    mask_out = make_square_img(sz_out, 1)
    r = 1 + bb_ext_offset
    sz_out2 = sz_out, sz_out
    img_sz = np.array((imgs[0].shape[1], imgs[0].shape[0]))
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

    for i in range(len(imgs)):
        if i > 0:
            img_patch.fill(0)
        img_patch[slices_patch] = imgs[i][slices_img]
        cv2.resize(img_patch, sz_out2, imgs_out[i], interpolation=cv2.INTER_AREA)

    mask_patch[slices_patch] = mask[slices_img]
    cv2.resize(mask_patch, sz_out2, mask_out, interpolation=cv2.INTER_NEAREST)

    bbox_ltwh_src = np.concatenate([bb_lt_fit, bb_sz_fit])
    ratio_src_to_dst = sz_out / bbox_sz_ext

    return imgs_out, mask_out, bbox_ltwh_src, ratio_src_to_dst


def img_to_tensor(img: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
    if mask is not None:
        img[~mask] = 0
    img = img.astype(np.float32) / 255
    return torch.from_numpy(img)


def imgs_list_to_tensors(imgs: list[np.ndarray], masks: Optional[list[np.ndarray]] = None) -> list[torch.Tensor]:
    res = []
    for i, img in enumerate(imgs):
        mask = None if masks is None else masks[i]
        res.append(img_to_tensor(img, mask))
    return res


def imgs_dict_to_tensors(imgs_dict: dict[str, list[np.ndarray]], masks: Optional[list[np.ndarray]] = None) -> dict[str, torch.Tensor]:
    return {k: imgs_list_to_tensors(imgs, masks) for k, imgs in imgs_dict.items()}


class GtImgsMasks:
    df_img: pd.DataFrame
    df_obj: pd.DataFrame
    imgs: ImgsList
    masks: ImgsList
    maps_names: list[str]
    maps: MapsDict
    imgs_crop: ImgsList
    masks_crop: ImgsList
    maps_crop: MapsDict
    bboxes_ltwh_src: list[np.ndarray]
    ratios_src_to_dst: list[float]
    imgs_crop_tn: list[torch.Tensor]
    maps_crop_tn: dict[str, torch.Tensor]

    def __init__(self, df_img: pd.DataFrame, df_obj: pd.DataFrame, imgs: ImgsList, masks: ImgsList, maps_names: list[str],
                 maps: MapsDict, imgs_crop: Optional[ImgsList] = None, masks_crop: Optional[ImgsList] = None,
                 maps_crop: Optional[MapsDict] = None, bboxes_ltwh_src: Optional[list[np.ndarray]] = None, ratios_src_to_dst: Optional[list[float]] = None,
                 imgs_crop_tn: Optional[list[torch.Tensor]] = None, maps_crop_tn: Optional[dict[str, torch.Tensor]] = None):
        self.df_img = df_img
        self.df_obj = df_obj
        self.imgs = imgs
        self.masks = masks
        self.maps_names = maps_names
        self.maps = maps
        self.imgs_crop = imgs_crop or []
        self.masks_crop = masks_crop or []
        self.maps_crop = maps_crop or {}
        self.bboxes_ltwh_src = bboxes_ltwh_src or []
        self.ratios_src_to_dst = ratios_src_to_dst or []
        self.imgs_crop_tn = imgs_crop_tn or []
        self.maps_crop_tn = maps_crop_tn or []

    def crop(self, patch_sz: int, offset: float = 0.05):
        img_id_to_ind = {self.df_img.index[i]: i for i in range(len(self.df_img))}
        map_names = list(self.maps.keys())
        imgs_crop = []
        maps_crop = {name: [] for name in map_names}
        masks_crop = []
        bboxes_ltwh_src = []
        ratios_src_to_dst = []
        for _, orow in self.df_obj.iterrows():
            img_ind = img_id_to_ind[orow['img_ds_id']]
            img = self.imgs[img_ind]
            maps = [self.maps[name][img_ind] for name in map_names]
            imgs = [img, *maps]
            imgs_out, mask_out, bbox_ltwh_src, ratio_src_to_dst = \
                crop_apply_mask(imgs, self.masks[img_ind], orow['bbox_visib_ltwh'], patch_sz, offset)
            imgs_crop.append(imgs_out[0])
            for i, name in enumerate(map_names):
                maps_crop[name].append(imgs_out[i + 1])
            masks_crop.append(mask_out)
            bboxes_ltwh_src.append(bbox_ltwh_src)
            ratios_src_to_dst.append(ratio_src_to_dst)
        self.imgs_crop = imgs_crop
        self.masks_crop = masks_crop
        self.maps_crop = maps_crop
        self.bboxes_ltwh_src = bboxes_ltwh_src
        self.ratios_src_to_dst = ratios_src_to_dst

    def gen_res(self, return_tensors: bool, keep_source_images: bool, keep_cropped_images: bool) -> 'GtImgsMasks':
        imgs, masks, maps = self.imgs, self.masks, self.maps
        imgs_crop, masks_crop, maps_crop = self.imgs_crop, self.masks_crop, self.maps_crop
        imgs_crop_tn, maps_crop_tn = [], []
        if return_tensors:
            assert len(imgs_crop) > 0
            imgs_crop_tn = imgs_list_to_tensors(imgs_crop, masks_crop)
            maps_crop_tn = imgs_dict_to_tensors(maps_crop, masks_crop)
        if not keep_source_images:
            imgs, masks, maps = [], [], []
        if not keep_cropped_images:
            imgs_crop, masks_crop, maps_crop = [], [], []
        return GtImgsMasks(self.df_img, self.df_obj, imgs, masks, self.maps_names, maps,
                           imgs_crop, masks_crop, maps_crop, self.bboxes_ltwh_src, self.ratios_src_to_dst,
                           imgs_crop_tn, maps_crop_tn)


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
    maps_names: list[str]
    out_size: Optional[int] = None
    aug_name: Optional[str] = None
    return_tensors: bool = False
    keep_source_images: bool = True
    keep_cropped_images: bool = True

    def __init__(self, ds_path: Path, irows: pd.DataFrame, orows: pd.DataFrame, out_size: Optional[int] = None,
                 aug_name: Optional[str] = None, maps_names: Optional[list[str]] = None, return_tensors: bool = False,
                 keep_source_images: bool = True, keep_cropped_images: bool = True):
        self.ds_path = ds_path
        self.irows = irows.copy()
        self.orows = orows.copy()
        self.out_size = out_size
        self.aug_name = aug_name
        self.maps_names = [] if maps_names is None else maps_names
        self.return_tensors = return_tensors
        self.keep_source_images = keep_source_images
        self.keep_cropped_images = keep_cropped_images


def load_imgs_objs_gt(params: ImgsObjsGtParams) -> GtImgsMasks:
    imgs, masks = [], []
    maps = {map_name: [] for map_name in params.maps_names}
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

        for map_name in params.maps_names:
            map_fpath = get_img_fpath(scene_path, img_id_str, map_name, 'png')
            map_ = imread(map_fpath)
            maps[map_name].append(map_)

    if params.aug_name is not None:
        aug = get_aug(params.aug_name)
        imgs = aug(images=imgs)

    gt = GtImgsMasks(params.irows, params.orows, imgs, masks, params.maps_names, maps)
    if params.out_size is not None:
        gt.crop(params.out_size)
    gt = gt.gen_res(params.return_tensors, params.keep_source_images, params.keep_cropped_images)

    return gt


class BopView:
    ds: 'BopDataset'
    ids: np.ndarray
    batch_size: Optional[int] = None
    aug_name: Optional[str] = None
    return_tensors: bool = False
    keep_source_images: bool = True
    keep_cropped_images: bool = True

    def __init__(self, ds: 'BopDataset', ids: np.ndarray, batch_size: Optional[int] = None, aug_name: Optional[str] = None,
                 return_tensors: Optional[bool] = None, keep_source_images: Optional[bool] = None, keep_cropped_images: Optional[bool] = None):
        self.ds = ds
        self.ids = ids.copy()
        self.batch_size = batch_size
        self.aug_name = aug_name
        self.return_tensors = self.return_tensors if return_tensors is None else return_tensors
        self.keep_source_images = self.keep_source_images if keep_source_images is None else keep_source_images
        self.keep_cropped_images = self.keep_cropped_images if keep_cropped_images is None else keep_cropped_images

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def shuffle(self):
        np.random.shuffle(self.ids)

    def __len__(self):
        return len(self.ids)

    def _get_img_obj_rows(self, inds: IndsList, is_ids: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, Optional[int]]:
        raise Exception('Unimplemented')

    def get_gt_task_params(self, i: IdsType, n: Optional[int] = None, is_ids: bool = False,
                           aug_name: Optional[str] = None, return_tensors: Optional[bool] = None,
                           keep_source_images: Optional[bool] = None, keep_cropped_images: Optional[bool] = None)\
            -> ImgsObjsGtParams:
        if type(i) in (list, np.ndarray, slice, range):
            assert n is None
            inds = i
        else:
            assert n is not None and n > 0
            inds = slice(i, i + n) if not is_ids else np.arange(i, i + n)
        if is_ids:
            inds = np.array(inds)
        irows, orows, out_size = self._get_img_obj_rows(inds, is_ids)
        return ImgsObjsGtParams(
            self.ds.get_ds_path(), irows, orows, out_size,
            aug_name and self.aug_name,
            self.ds.maps_names,
            self.return_tensors if return_tensors is None else return_tensors,
            self.keep_source_images if keep_source_images is None else keep_source_images,
            self.keep_cropped_images if keep_cropped_images is None else keep_cropped_images,
        )

    def get_gt_imgs_masks(self, i: IdsType, n: Optional[int] = None, is_ids: bool = False) -> GtImgsMasks:
        params = self.get_gt_task_params(i, n, is_ids)
        return load_imgs_objs_gt(params)

    def get_gt_batch(self, i_batch: int) -> GtImgsMasks:
        assert self.batch_size is not None
        i = i_batch * self.batch_size
        return self.get_gt_imgs_masks(i, self.batch_size)

    def get_batch_iterator(self, n_batches: Optional[int] = None, multiprocess: bool = False, batch_size: Optional[int] = None,
                           drop_last: bool = False, shuffle_between_loops: bool = True, buffer_sz: int = 3, aug_name: Optional[str] = None,
                           return_tensors: Optional[bool] = None, keep_source_images: Optional[bool] = None, keep_cropped_images: Optional[bool] = None)\
            -> Generator[GtImgsMasks, None, None]:
        batch_size = batch_size or self.batch_size
        n = len(self.ids)
        n_batches_total = n // batch_size + min(n % batch_size, 1)

        info = f'n = {n}. batch_size = {batch_size}. n_batches = {n_batches}. n_batches_total = {n_batches_total}'
        assert n_batches_total > 0, info
        assert n_batches is None or n_batches > 0, info

        looped = False
        if n_batches is None:
            n_batches = n_batches_total
        if n_batches > n_batches_total:
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
                params = self.get_gt_task_params(
                    inds, aug_name=aug_name, return_tensors=return_tensors,
                    keep_source_images=keep_source_images, keep_cropped_images=keep_cropped_images)
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
    def __init__(self, ds: 'BopDataset', ids: np.ndarray, batch_size: Optional[int] = None, aug_name: Optional[str] = None,
                 return_tensors: Optional[bool] = None, keep_source_images: Optional[bool] = None, keep_cropped_images: Optional[bool] = None):
        super().__init__(ds, ids, batch_size, aug_name, return_tensors, keep_source_images, keep_cropped_images)

    def _get_img_obj_rows(self, inds: IndsList, is_ids: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, Optional[int]]:
        img_ds_ids = inds if is_ids else self.ids[inds]
        dfi, dfo = self.ds.df_img, self.ds.df_obj
        irows = dfi.loc[img_ds_ids]
        orows = dfo[dfo['img_ds_id'].isin(img_ds_ids)]
        return irows, orows, None


class BopObjsView(BopView):
    obj_ids: np.ndarray
    out_size: Optional[int] = None
    min_bbox_dim_ratio: Optional[float] = None
    min_mask_ratio: Optional[float] = None

    def __init__(self, ds: 'BopDataset', ids: Optional[np.ndarray] = None, obj_ids: Optional[IdsType] = None,
                 batch_size: Optional[int] = None, out_size: Optional[int] = None, min_bbox_dim_ratio: Optional[float] = None,
                 min_mask_ratio: Optional[float] = None, aug_name: Optional[str] = None,
                 return_tensors: Optional[bool] = None, keep_source_images: Optional[bool] = None, keep_cropped_images: Optional[bool] = None):
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
            dfo = dfo[dfo['px_count_visib'] > 1e-6]
            dfi = ds.df_img.loc[dfo['img_ds_id']]
            if min_mask_ratio is not None:
                img_area = dfi['img_width'] * dfi['img_height']
                inds = dfo['px_count_visib'].values / img_area.values >= min_mask_ratio
                print(f'min_mask_ratio: {len(inds)} --> {inds.sum()}')
                dfo, dfi = dfo[inds], dfi[inds]
            if min_bbox_dim_ratio is not None:
                bbox_ltwh = np.stack(dfo['bbox_visib_ltwh'])
                bbox_sz = bbox_ltwh[:, 2:].min(axis=1)
                img_area = dfi[['img_width', 'img_height']].max(axis=1)
                inds = bbox_sz / img_area.values >= min_bbox_dim_ratio
                print(f'min_bbox_dim_ratio: {len(inds)} --> {inds.sum()}')
                dfo = dfo[inds]
            ids = dfo.index.values

        self.min_mask_ratio = min_mask_ratio
        self.min_bbox_dim_ratio = min_bbox_dim_ratio
        self.out_size = out_size
        super().__init__(ds, ids, batch_size, aug_name, return_tensors, keep_source_images, keep_cropped_images)
        self.obj_ids = obj_ids

    def set_out_size(self, out_size: int):
        self.out_size = out_size

    def _get_img_obj_rows(self, inds: IndsList, is_ids: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, Optional[int]]:
        obj_ds_ids = inds if is_ids else self.ids[inds]
        orows = self.ds.df_obj.loc[obj_ds_ids]
        img_ds_ids = pd.unique(orows['img_ds_id'])
        irows = self.ds.df_img.loc[img_ds_ids]
        return irows, orows, self.out_size


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
    maps_names: list[str]

    pool: Optional[mpr.Pool] = None
    pool_refcount = 0

    def __init__(self, bop_path: Path, ds_name: str, ds_subdir: str, df_img: pd.DataFrame, df_obj: pd.DataFrame, shuffled: bool,
                 maps_names: tuple[str] = ('norm', 'noc')) -> None:
        self.bop_path = bop_path
        self.ds_name = ds_name
        self.ds_subdir = ds_subdir
        self.df_img = df_img
        self.df_obj = df_obj
        self.shuffled = shuffled
        self.maps_names = list(maps_names)
    
    @classmethod
    def get_cache_path(cls, bop_path: Path, ds_name: str) -> Path:
        return bop_path / ds_name / cls.cache_subdir

    # Returns if cache was written
    def shuffle(self) -> bool:
        if self.shuffled:
            return False
        print('Shuffle!')
        self.df_img = self.df_img.sample(len(self.df_img))
        self.df_obj = self.df_obj.sample(len(self.df_obj))
        self.shuffled = True
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
    def from_dir(cls, bop_path: Path, ds_name: str, ds_subdir: str, shuffle: bool, skip_cache: bool = False,
                 maps_names: tuple[str] = ('norm', 'noc')) -> 'BopDataset':
        ds_path = bop_path / ds_name
        train_path = ds_path / ds_subdir
        cache_path = cls.get_cache_path(bop_path, ds_name)
        if skip_cache:
            print(f'Removing {cache_path}')
            shutil.rmtree(cache_path, ignore_errors=True)
        
        if cache_path.exists():
            ds = cls.read_cache(cache_path)
            if ds is not None:
                if shuffle:
                    ds.shuffle()
                ds.maps_names = maps_names
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

        ds = BopDataset(bop_path, ds_name, ds_subdir, df_img, df_obj, shuffled=False, maps_names=maps_names)
        written_to_cache = False
        if shuffle:
            written_to_cache = ds.shuffle()
        if not written_to_cache:
            ds.write_cache()

        return ds

    def get_imgs_view(self, batch_size: Optional[int] = None, aug_name: Optional[str] = None,
                      return_tensors: Optional[bool] = None, keep_source_images: Optional[bool] = None, keep_cropped_images: Optional[bool] = None
                      ) -> BopImgsView:
        return BopImgsView(
            self, self.df_img.index.values, batch_size, aug_name=aug_name,
            return_tensors=return_tensors, keep_source_images=keep_source_images, keep_cropped_images=keep_cropped_images,
        )

    def get_objs_view(self, obj_ids: IdsType = -1, batch_size: Optional[int] = None,
                      out_size: Optional[int] = None, min_bbox_dim_ratio: Optional[float] = 0.05,
                      min_mask_ratio: Optional[float] = 0.001, aug_name: Optional[str] = None,
                      return_tensors: Optional[bool] = None, keep_source_images: Optional[bool] = None, keep_cropped_images: Optional[bool] = None
                      ) -> BopObjsView:
        return BopObjsView(
            self, obj_ids=obj_ids, batch_size=batch_size, out_size=out_size, min_bbox_dim_ratio=min_bbox_dim_ratio,
            min_mask_ratio=min_mask_ratio, aug_name=aug_name,
            return_tensors=return_tensors, keep_source_images=keep_source_images, keep_cropped_images=keep_cropped_images,
        )

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

