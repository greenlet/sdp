import numpy as np
import os
import torch
from enum import Enum
from pydantic import Field
from pydantic_cli import run_and_exit
from tqdm import trange
from typing import Optional

from sdp.ds.bop_dataset import BopDataset, BopEpochIterator
from sdp.ds.emb_dataset import EmbDataset
from sdp.models.segmenter.factory import create_vit
from sdp.utils.tensor import stack_imgs_maps
from sdp.utils.train import ArgsAaeBase, ConfigAaeTrain


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


class ArgsAaeRun(ArgsAaeBase):
    train_subdir: str = Field(
        ...,
        required=True,
        description=f'Train subdirectory of the format "ds_<dataset-name>_obj_<obj-id>_imgsz_<img-size>_<start-datetime-YYYYmmdd_HHMMSS>".',
        cli=('--train-subdir',),
    )


def main(args: ArgsAaeRun) -> int:
    print(args)
    train_path = args.train_root_path / args.train_subdir
    assert train_path.exists(), f'Train path "{train_path}" does not exist'
    print(f'Reading train config from {train_path / ConfigAaeTrain._fname}')
    tcfg = ConfigAaeTrain.from_yaml(train_path)
    print(tcfg)

    ds_path = tcfg.bop_root_path / tcfg.dataset_name
    print(f'ds_path: {ds_path}')
    ds = BopDataset.from_dir(tcfg.bop_root_path, tcfg.dataset_name, shuffle=False)
    objs_view = ds.get_objs_view(tcfg.obj_id, args.eval_batch_size, tcfg.img_size)
    device = torch.device(args.device)
    print(f'Pytorch device: {device}')

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
    }
    encoder = create_vit(model_cfg)
    encoder.to(device)
    # print(encoder)
    print(f'Loading model from {tcfg.best_checkpoint_fpath}')
    checkpoint = torch.load(tcfg.best_checkpoint_fpath)
    print(f'Checkpoint keys: {list(checkpoint.keys())}')
    encoder.load_state_dict(checkpoint['model'], strict=False)

    n_items = len(objs_view)
    n_batches = objs_view.n_batches()
    # n_batches = 10
    # val_it = objs_view.get_batch_iterator(
    #     n_batches=n_batches,
    #     batch_size=tcfg.eval_batch_size,
    #     multiprocess=args.ds_mp_loading,
    #     mp_queue_size=args.eval_mp_queue_size,
    #     return_tensors=True,
    #     keep_source_images=False,
    #     keep_cropped_images=False,
    #     drop_last=False,
    # )
    val_epoch_it = BopEpochIterator(
        bop_view=objs_view, n_epochs=1, n_batches_per_epoch=n_batches, batch_size=tcfg.eval_batch_size,
        drop_last=False, shuffle_between_loops=False, multiprocess=args.ds_mp_loading,
        mp_pool_size=args.ds_mp_pool_size, mp_queue_size=args.eval_mp_queue_size,
        return_tensors=False, keep_source_images=False, keep_cropped_images=True,
    )
    pbar = trange(n_batches, desc=f'Eval', unit='batch')
    encoder.eval()

    emb_sz = encoder.n_cls
    n_items = min(n_batches * tcfg.eval_batch_size, len(objs_view))
    obj_ds_ids = np.empty(n_items, int)
    embs = np.empty((n_items, emb_sz), np.float32)

    off = 0
    val_it = iter(val_epoch_it.get_batch_iterator())
    for step in pbar:
        gt_item = next(val_it)
        gt_item.to_tensors()
        x = stack_imgs_maps(gt_item.maps_crop_tn, gt_item.maps_names, gt_item.imgs_crop_tn)
        x = x.to(device)
        y = encoder.forward(x)
        y = y.detach().to('cpu')
        # print(y.shape, np.linalg.norm(y[0] - y[1]), np.linalg.norm(y[0] - y[2]))
        inds = slice(off, min(off + y.shape[0], obj_ds_ids.shape[0]))
        obj_ds_ids[inds] = gt_item.df_obj.index
        embs[inds] = y
        off += y.shape[0]

    assert set(objs_view.ids[:n_batches * tcfg.eval_batch_size]) == set(obj_ds_ids)

    pbar.close()
    del val_it

    out_path = ds.get_ds_path() / tcfg.train_subdir
    os.makedirs(out_path, exist_ok=True)
    tcfg.to_yaml(out_path)

    cache_path = ds.get_cache_path()
    emb_data_path = cache_path / tcfg.train_subdir
    emb_ds = EmbDataset(emb_data_path, tcfg, obj_ds_ids, embs)
    emb_ds.write_cache()

    # emb_ds1 = EmbDataset.read_cache(emb_data_path)
    # print('embs1:', embs.shape, embs.dtype)
    # print('ids1:', obj_ds_ids.shape, obj_ds_ids.dtype)
    # print('embs2:', emb_ds1.embs.shape, emb_ds1.embs.dtype)
    # print('ids2:', emb_ds1.obj_ds_ids.shape, emb_ds1.obj_ds_ids.dtype)
    # print('embs close:', np.allclose(embs, emb_ds1.embs))
    # print('ids close:', np.allclose(obj_ds_ids, emb_ds1.obj_ds_ids))

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsAaeRun, main, 'Run inference for Augmented Auto Encoder for single object.')

