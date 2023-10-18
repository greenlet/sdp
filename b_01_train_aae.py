import argparse
import math
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, validator
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str, parse_yaml_file_as
from tqdm import trange

from sdp.ds.bop_dataset import BopDataset, AUGNAME_DEFAULT
from sdp.models.segmenter.factory import create_segmenter
from sdp.utils.tensor import stack_imgs_maps
from segm.optim.factory import create_optimizer, create_scheduler


class TrainSubdirType(str, Enum):
    New = 'new'
    Last = 'last'
    LastOrNew = 'last_or_new'

    @classmethod
    def from_str(cls, val: str) -> Optional['TrainSubdirType']:
        val = val.lower()
        if val in list(cls):
            return TrainSubdirType(val)
        return None


class Config(BaseModel):
    bop_root_path: Path = Field(
        ...,
        required=True,
        description='Path to BOP datasets (containing datasets: itodd, tless, etc.)',
        cli=('--bop-root-path',),
    )
    train_root_path: Path = Field(
        ...,
        required=True,
        description='Path to train root directory. New train processes will create subdirectories in it.',
        cli=('--train-root-path',),
    )
    dataset_name: Optional[str] = Field(
        None,
        required=False,
        description='Dataset name. Has to be a subdirectory of BOP_ROOT_PATH, one of: "itodd", "tless", etc.',
        cli=('--dataset-name',),
    )
    obj_id: Optional[int] = Field(
        None,
        required=False,
        description='BOP object id. Can have values: 1, 2, ..., n. n - maximum obj id number for DATASET_NAME.',
        cli=('--obj-id',)
    )
    img_size: Optional[int] = Field(
        None,
        required=False,
        description='Encoder input and Decoder output image size.',
        cli=('--img-size',),
    )
    epochs: Optional[int] = Field(
        None,
        required=False,
        description='Number of training epochs.',
        cli=('--epochs',),
    )
    batch_size: Optional[int] = Field(
        None,
        required=False,
        description='Training batch size (number of images for single forward/backward network iteration).',
        cli=('--batch-size',),
    )
    train_subdir: Optional[str] = Field(
        'last_or_new',
        required=False,
        description=f''
        f'Train subdirectory reference. It can have string value of the '
        f'FORMAT="ds_<dataset-name>_obj_<obj-id>_imgsz_<img-size>_<start-datetime-YYYYmmdd_HHMMSS>". '
        f'In this case its value is considered TRAIN_ROOT_PATH subdirectory. Other values it can have: '
        f'- "{TrainSubdirType.New}". New subdirectory of the FORMAT will be generated; '
        f'- "{TrainSubdirType.Last}". Subdirectory of the FORMAT with latest datetime will be searched for.'
        f'Error is raised if none found; '
        f'- "{TrainSubdirType.LastOrNew}". Latest subdirectory of the FORMAT will be searched for. '
        f'New one is generated if none found; for the values "new", "last", "last_or_new" DATASET_NAME, OBJ_ID, '
        f'IMG_SIZE must be set.',
        cli=('--train-subdir',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run training on. Can have values: "cpu", "gpu"',
        cli=('--device',)
    )


TRAIN_SUBDIR_PAT = re.compile(r'^ds_(?P<ds_name>\w+)_obj_(?P<obj_id>\d+)_imsz_(?P<img_sz>\d+)_'
                              r'(?P<dt_str>\d{8}_\d{6})$')
DATE_PAT = '%Y%m%d_%H%M%S'


def format_date(dt: datetime) -> str:
    return dt.strftime(DATE_PAT)


def parse_date(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, DATE_PAT)


def format_train_subdir(ds_name: str, obj_id: int, img_sz: int, dt: datetime) -> str:
    return f'ds_{ds_name}_obj_{obj_id}_imsz_{img_sz}_{format_date(dt)}'


def parse_train_subdir(subdir: str, silent: bool = False) -> Optional[tuple[str, int, int, datetime]]:
    m = TRAIN_SUBDIR_PAT.match(subdir)
    if not m:
        if silent:
            return
        raise Exception(f'Cannot parse string "{subdir}" with pattern "{TRAIN_SUBDIR_PAT}"')
    ds_name = m.group('ds_name')
    obj_id = int(m.group('obj_id'))
    img_sz = int(m.group('img_sz'))
    dt_str = m.group('dt_str')
    try:
        dt = parse_date(dt_str)
    except:
        if silent:
            return
        raise Exception(f'Cannot parse date "{dt_str}" in "{subdir}" with pattern "{DATE_PAT}"')
    return ds_name, obj_id, img_sz, dt


class TrainCfg(BaseModel):
    _fname: str = 'train_config.yaml'
    _best_checkpoint_fname: str = 'best.pth'
    _last_checkpoint_fname: str = 'last.pth'
    bop_root_path: Path
    train_root_path: Path
    train_subdir: str
    dataset_name: str
    obj_id: int
    dt: datetime
    img_size: int
    last_epoch: int
    epochs: int
    batch_size: int
    train_path: Path
    last_checkpoint_fpath: Path
    best_checkpoint_fpath: Path

    def __init__(self, **kwargs):
        kwargs['bop_root_path'] = Path(kwargs['bop_root_path'])
        kwargs['train_root_path'] = Path(kwargs['train_root_path'])
        train_path = kwargs['train_root_path'] / kwargs['train_subdir']
        kwargs['train_path'] = train_path
        kwargs['last_checkpoint_fpath'] = train_path / self._last_checkpoint_fname
        kwargs['best_checkpoint_fpath'] = train_path / self._best_checkpoint_fname
        super().__init__(**kwargs)

    @classmethod
    def from_yaml(cls, fpath: Optional[Path] = None) -> 'TrainCfg':
        if fpath.is_dir():
            fpath = fpath / cls._fname
        return parse_yaml_file_as(TrainCfg, fpath)

    def to_yaml(self, fpath: Optional[Path] = None, overwrite: bool = True):
        if fpath is None:
            fpath = self.train_path / self._fname
        elif fpath.is_dir():
            fpath /= self._fname
        if fpath.exists() and not overwrite:
            return
        with open(fpath, 'w') as f:
            f.write(to_yaml_str(self, indent=2))

    def create_paths(self):
        self.train_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _empty_in_cfg(cls, cfg: Config, must_attrs: list[str]) -> str:
        err_attrs = []
        for attr in must_attrs:
            if not hasattr(cfg, attr) or getattr(cfg, attr) is None:
                err_attrs.append(attr)
        if err_attrs:
            return ', '.join(err_attrs)
        return ''

    @classmethod
    def _check_attrs(cls, cfg: Config, reason: str, *attrs: str):
        err_attrs = cls._empty_in_cfg(cfg, list(attrs))
        if err_attrs:
            err_str = f'When {reason} following parameters must be set: {err_attrs}'
            raise Exception(err_str)

    @classmethod
    def _check_attrs_short(cls, cfg: Config, reason: str):
        cls._check_attrs(cfg, reason, 'dataset_name', 'obj_id', 'img_size')

    @classmethod
    def _check_attrs_full(cls, cfg: Config, reason: str):
        cls._check_attrs(cfg, reason, 'dataset_name', 'obj_id', 'img_size', 'epochs', 'batch_size')

    class _TrainSubdir(BaseModel):
        ds_name: str
        obj_id: int
        img_sz: int
        dt: datetime
        name: str

        def __init__(self, **kwargs):
            if 'dt' not in kwargs:
                kwargs['dt'] = datetime.now()
            kwargs['name'] = format_train_subdir(**kwargs)
            super().__init__(**kwargs)

    @classmethod
    def _find_last_train_subdir(cls, cfg: Config) -> Optional[_TrainSubdir]:
        last_subdir: Optional[cls._TrainSubdir] = None
        for tpath in cfg.train_root_path.iterdir():
            config_fpath = tpath / cls._fname
            if not tpath.is_dir() or not config_fpath.exists():
                continue
            subdir_params = parse_train_subdir(tpath.name, silent=True)
            if not subdir_params:
                continue
            ds_name, obj_id, img_sz, dt = subdir_params
            if last_subdir is None or last_subdir.dt < dt:
                last_subdir = cls._TrainSubdir(ds_name=ds_name, obj_id=obj_id, img_sz=img_sz, dt=dt)
        return last_subdir

    @classmethod
    def from_cfg(cls, cfg: Config) -> 'TrainCfg':
        train_subdir_src = TrainSubdirType.from_str(cfg.train_subdir)
        train_subdir: Optional[cls._TrainSubdir] = None
        if train_subdir_src == TrainSubdirType.New:
            cls._check_attrs_full(cfg, f'train_subdir = {train_subdir_src}')
            train_subdir = cls._TrainSubdir(ds_name=cfg.dataset_name, obj_id=cfg.obj_id, img_sz=cfg.img_size)
        elif train_subdir_src == TrainSubdirType.Last:
            cls._check_attrs_short(cfg, f'train_subdir = {train_subdir_src}')
            train_subdir = cls._find_last_train_subdir(cfg)
            if train_subdir is None:
                raise Exception(f'Cannot find last subdir in {cfg.train_root_path} for cfg: {cfg}')
        elif train_subdir_src == TrainSubdirType.LastOrNew:
            cls._check_attrs_short(cfg, f'train_subdir = {train_subdir_src}')
            train_subdir = cls._find_last_train_subdir(cfg)
            if train_subdir is None:
                cls._check_attrs_full(cfg, f'last subdir not found for train_subdir = {train_subdir_src}')
                train_subdir = cls._TrainSubdir(ds_name=cfg.dataset_name, obj_id=cfg.obj_id, img_sz=cfg.img_size)
        else:
            tpath = cfg.train_root_path / cfg.train_subdir
            if not tpath.exists():
                raise Exception(f'Directory {tpath} does not exist')
            fpath = tpath / cls._fname
            if not fpath.exists():
                raise Exception(f'Config file {fpath} does not exist')
            ds_name, obj_id, img_sz, dt = parse_train_subdir(cfg.train_subdir)
            tcfg = cls.from_yaml(fpath)
            assert ds_name == tcfg.dataset_name and obj_id == tcfg.obj_id and img_sz == tcfg.img_size and dt == tcfg.dt,\
                f'Train subdir "{train_subdir}" does not match config parameters: {tcfg}'
            return tcfg

        tpath = cfg.train_root_path / train_subdir.name
        fpath = tpath / cls._fname
        if fpath.exists():
            return cls.from_yaml(fpath)

        return TrainCfg(
            bop_root_path=cfg.bop_root_path,
            train_root_path = cfg.train_root_path,
            train_subdir=train_subdir.name,
            dataset_name=cfg.dataset_name,
            obj_id=cfg.obj_id,
            dt=train_subdir.dt,
            img_size=cfg.img_size,
            last_epoch=0,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size
        )


def main(cfg: Config) -> int:
    cfg.train_root_path.mkdir(parents=True, exist_ok=True)
    tcfg = TrainCfg.from_cfg(cfg)
    print('Config:', cfg)
    print('Train Config:', tcfg)
    tcfg.create_paths()
    tcfg.to_yaml(overwrite=False)

    device = torch.device(cfg.device)
    skip_cache = False
    ds = BopDataset.from_dir(tcfg.bop_root_path, tcfg.dataset_name, 'train_pbr', shuffle=True, skip_cache=skip_cache)
    ds.shuffle()
    objs_view = ds.get_objs_view(tcfg.obj_id, tcfg.batch_size, tcfg.img_size, return_tensors=True,
                                 keep_source_images=False, keep_cropped_images=False)
    ov_train, ov_val = objs_view.split((-1, 0.1))
    ov_train.set_aug_name(AUGNAME_DEFAULT)
    it_buffer_sz = 10

    inp_ch, n_cls = 9, 6
    backbone = 'vit_tiny_patch16_384'
    model_cfg = {
        'n_cls': n_cls,
        'backbone': backbone,
        'image_size': (tcfg.img_size, tcfg.img_size),
        'channels': inp_ch,
        'patch_size': 16,
        'd_model': 192,
        'n_heads': 3,
        'n_layers': 12,
        'normalization': 'vit',
        'distilled': False,
        'decoder': {
            'name': 'mask_transformer',
            'drop_path_rate': 0.1,
            'dropout': 0.1,
            'n_layers': 2
        }
    }
    opt_cfg = {
        'opt': 'sgd',
        'lr': 0.001,
        'weight_decay': 0.0,
        'momentum': 0.9,
        'clip_grad': None,
        'sched': 'polynomial',
        'epochs': tcfg.epochs,
        'min_lr': 1e-5,
        'poly_power': 0.9,
        'poly_step_size': 1,
        'iter_max': tcfg.epochs * len(ov_train),
        'iter_warmup': 0.0,
    }
    model = create_segmenter(model_cfg)
    model.to(device)
    # print(model)

    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in opt_cfg.items():
        opt_vars[k] = v

    optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    if cfg.train_subdir == 'last' and tcfg.last_checkpoint_fpath.exists():
        checkpoint = torch.load(tcfg.last_checkpoint_fpath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        tcfg.last_epoch = checkpoint['last_epoch']

    num_updates = 0
    val_loss = None
    for epoch in range(tcfg.last_epoch + 1, tcfg.epochs + 1):
        train_it = ov_train.get_batch_iterator(
            batch_size=tcfg.batch_size,
            shuffle_between_loops=True,
            multiprocess=True,
            buffer_sz=it_buffer_sz,
            return_tensors=True,
            keep_source_images=False,
            keep_cropped_images=False,
        )
        pbar = trange(len(ov_train), desc=f'Epoch {epoch}', unit='batch')
        model.train()
        for step, gt_item in zip(iter(pbar), train_it):
            x = stack_imgs_maps(gt_item.maps_crop_tn, gt_item.maps_names, gt_item.imgs_crop_tn)
            y_gt = stack_imgs_maps(gt_item.maps_crop_tn, gt_item.maps_names)
            x, y_gt = x.to(device), y_gt.to(device)
            y_pred = model.forward(x)
            loss: torch.Tensor = criterion(y_pred, y_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_updates += 1
            lr_scheduler.step_update(num_updates=num_updates)
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(f'Train. loss: {loss.item():.4f}. lr: {lr:.6f}')

        pbar.close()

        val_it = ov_val.get_batch_iterator(
            batch_size=tcfg.batch_size,
            shuffle_between_loops=True,
            multiprocess=True,
            buffer_sz=it_buffer_sz,
            return_tensors=True,
            keep_source_images=False,
            keep_cropped_images=False,
        )
        pbar = trange(len(ov_train), desc=f'Epoch {epoch}. Val', unit='batch')
        model.eval()
        loss_avg = 0
        for step, gt_item in zip(iter(pbar), val_it):
            x = stack_imgs_maps(gt_item.maps_crop_tn, gt_item.maps_names, gt_item.imgs_crop_tn)
            y_gt = stack_imgs_maps(gt_item.maps_crop_tn, gt_item.maps_names)
            y_pred = model.forward(x)
            loss: torch.Tensor = criterion(y_pred, y_gt)
            pbar.set_postfix_str(f'Val. loss: {loss.item():.4f}')
            loss_avg += loss.item()
        pbar.close()
        loss_avg /= len(ov_val)
        print(f'Val loss avg: {loss_avg:.4f}')

        best = False
        if val_loss is None or loss_avg < val_loss:
            val_loss = loss_avg
            best = True

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': lr_scheduler.state_dict(),
            'last_epoch': epoch,
            'best_val_loss': val_loss,
        }
        print(f'Saving checkpoint in {tcfg.last_checkpoint_fpath}')
        torch.save(checkpoint, tcfg.last_checkpoint_fpath)

        if best:
            print(f'New val loss minimum: {loss_avg:.4f}. Saving checkpoint to {tcfg.best_checkpoint_fpath}')
            shutil.copyfile(tcfg.last_checkpoint_fpath, tcfg.best_checkpoint_fpath)

    return 0


if __name__ == '__main__':
    run_and_exit(Config, main, 'Train Augmented Auto Encoder for single object.')


