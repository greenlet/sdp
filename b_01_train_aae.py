import argparse
from contextlib import suppress
import re
import shutil
import torch
import torch.utils.tensorboard as tb
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from tqdm import trange
from typing import Optional, Any

from sdp.ds.bop_dataset import BopDataset, AUGNAME_DEFAULT, BopEpochIterator
from sdp.models.segmenter.factory import create_segmenter
from sdp.utils.tensor import stack_imgs_maps
from sdp.utils.train import MeanWin, ArgsAaeBase, ConfigAaeTrain, imgs_dict_to_tensors, imgs_list_to_tensors
from segm.optim.factory import create_optimizer, create_scheduler
from timm.utils import NativeScaler


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


class ArgsAaeTrain(ArgsAaeBase):
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
    train_batch_size: Optional[int] = Field(
        None,
        required=False,
        description='Training batch size (number of images for single training forward/backward network run).',
        cli=('--train-batch-size',),
    )
    train_mp_queue_size: int = Field(
        3,
        required=False,
        description='Train dataset multiprocess data loader queue size.',
        cli=('--train-mp-queue-size',)
    )
    epochs: Optional[int] = Field(
        None,
        required=False,
        description='Number of training epochs.',
        cli=('--epochs',),
    )
    train_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of training steps per epoch. When TRAIN_EPOCH_STEPS <= 0 '
                    'the number of steps will be a number of batches contained in dataset. (default is -1)',
        cli=('--train-epoch-steps',),
    )
    val_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of validation steps per epoch. When VAL_EPOCH_STEPS <= 0 '
                    'the number of steps will be a number of batches contained in dataset. (default is -1)',
        cli=('--val-epoch-steps',),
    )
    train_subdir: str = Field(
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
    learning_rate: float = Field(
        0.001,
        required=False,
        description='Initial learning rate of the training process.',
        cli=('--learning-rate',)
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


def empty_in_cfg(args: ArgsAaeTrain, must_attrs: list[str]) -> str:
    err_attrs = []
    for attr in must_attrs:
        if not hasattr(args, attr) or getattr(args, attr) is None:
            err_attrs.append(attr)
    if err_attrs:
        return ', '.join(err_attrs)
    return ''


def check_attrs(args: ArgsAaeTrain, reason: str, *attrs: str):
    err_attrs = empty_in_cfg(args, list(attrs))
    if err_attrs:
        err_str = f'When {reason} following parameters must be set: {err_attrs}'
        raise Exception(err_str)


def check_attrs_short(args: ArgsAaeTrain, reason: str):
    check_attrs(args, reason, 'dataset_name', 'obj_id', 'img_size')


def check_attrs_full(args: ArgsAaeTrain, reason: str):
    check_attrs(args, reason, 'dataset_name', 'obj_id', 'img_size', 'epochs', 'train_batch_size')


class TrainSubdir(BaseModel):
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


def find_last_train_subdir(args: ArgsAaeTrain) -> Optional[TrainSubdir]:
    last_subdir: Optional[TrainSubdir] = None
    for tpath in args.train_root_path.iterdir():
        config_fpath = tpath / ConfigAaeTrain._fname
        if not tpath.is_dir() or not config_fpath.exists():
            continue
        subdir_params = parse_train_subdir(tpath.name, silent=True)
        if not subdir_params:
            continue
        ds_name, obj_id, img_sz, dt = subdir_params
        if last_subdir is None or last_subdir.dt < dt:
            last_subdir = TrainSubdir(ds_name=ds_name, obj_id=obj_id, img_sz=img_sz, dt=dt)
    return last_subdir


def coalesce(val: Optional[Any], default: Any):
    return val if val is not None else default


def update_tcfg(tcfg: 'ConfigAaeTrain', args: ArgsAaeTrain):
    tcfg.epochs = coalesce(args.epochs, tcfg.epochs)
    tcfg.train_epoch_steps = coalesce(args.train_epoch_steps, tcfg.train_epoch_steps)
    tcfg.val_epoch_steps = coalesce(args.val_epoch_steps, tcfg.val_epoch_steps)
    tcfg.train_batch_size = coalesce(args.train_batch_size, tcfg.train_batch_size)
    tcfg.eval_batch_size = coalesce(args.eval_batch_size, tcfg.eval_batch_size)


def tcfg_from_args(args: ArgsAaeTrain) -> 'ConfigAaeTrain':
    train_subdir_src = TrainSubdirType.from_str(args.train_subdir)
    train_subdir: Optional[TrainSubdir] = None
    if train_subdir_src == TrainSubdirType.New:
        check_attrs_full(args, f'train_subdir = {train_subdir_src}')
        train_subdir = TrainSubdir(ds_name=args.dataset_name, obj_id=args.obj_id, img_sz=args.img_size)
    elif train_subdir_src == TrainSubdirType.Last:
        check_attrs_short(args, f'train_subdir = {train_subdir_src}')
        train_subdir = find_last_train_subdir(args)
        if train_subdir is None:
            raise Exception(f'Cannot find last subdir in {args.train_root_path} for cfg: {args}')
    elif train_subdir_src == TrainSubdirType.LastOrNew:
        check_attrs_short(args, f'train_subdir = {train_subdir_src}')
        train_subdir = find_last_train_subdir(args)
        if train_subdir is None:
            check_attrs_full(args, f'last subdir not found for train_subdir = {train_subdir_src}')
            train_subdir = TrainSubdir(ds_name=args.dataset_name, obj_id=args.obj_id, img_sz=args.img_size)
    else:
        tpath = args.train_root_path / args.train_subdir
        if not tpath.exists():
            raise Exception(f'Directory {tpath} does not exist')
        fpath = tpath / ConfigAaeTrain._fname
        if not fpath.exists():
            raise Exception(f'Config file {fpath} does not exist')
        ds_name, obj_id, img_sz, dt = parse_train_subdir(args.train_subdir)
        tcfg = ConfigAaeTrain.from_yaml(fpath)
        assert ds_name == tcfg.dataset_name and obj_id == tcfg.obj_id and img_sz == tcfg.img_size and dt == tcfg.dt,\
            f'Train subdir "{train_subdir}" does not match config parameters: {tcfg}'

        update_tcfg(tcfg, args)
        return tcfg

    tpath = args.train_root_path / train_subdir.name
    fpath = tpath / ConfigAaeTrain._fname
    if fpath.exists():
        tcfg = ConfigAaeTrain.from_yaml(fpath)
        update_tcfg(tcfg, args)
        return tcfg

    tcfg = ConfigAaeTrain(
        bop_root_path=args.bop_root_path,
        train_root_path=args.train_root_path,
        train_subdir=train_subdir.name,
        dataset_name=args.dataset_name,
        obj_id=args.obj_id,
        dt=train_subdir.dt,
        img_size=args.img_size,
        last_epoch=0,
        epochs=args.epochs,
        train_epoch_steps=coalesce(args.train_epoch_steps, -1),
        val_epoch_steps=coalesce(args.val_epoch_steps, -1),
        train_batch_size=args.train_batch_size,
        eval_batch_size=coalesce(args.eval_batch_size, args.train_batch_size),
        learning_rate=args.learning_rate,
    )
    return tcfg


def tile_images(imgs_gt: torch.Tensor, imgs_pred: torch.Tensor, n_maps: int, max_imgs: int = 5) -> torch.Tensor:
    img_sz = imgs_gt.shape[-1]
    desc = f'imgs_gt.shape = {imgs_gt.shape}. imgs_pred.shape = {imgs_pred.shape}. n_maps = {n_maps}, max_imgs = {max_imgs}'
    assert imgs_gt.shape[-2] == img_sz, desc
    assert imgs_gt.shape[-3] // 3 == n_maps + 1, desc
    assert imgs_pred.shape[-2] == imgs_pred.shape[-1] == img_sz, desc
    assert imgs_pred.shape[-3] // 3 == n_maps, desc

    nr, nc = min(imgs_gt.shape[0], max_imgs), 2 * n_maps + 1
    height, width = nr * img_sz, nc * img_sz

    slim = lambda ind: slice(ind * img_sz, (ind + 1) * img_sz)
    slch = lambda ind: slice(ind * 3, (ind + 1) * 3)
    res = torch.zeros(3, height, width, dtype=torch.float32)
    for i in range(nr):
        v_slice = slim(i)
        # print(res[:, v_slice, slim(0)].shape, imgs_gt[i, slch(0)].shape)
        res[:, v_slice, slim(0)] = imgs_gt[i, slch(0)]
        for i_map in range(n_maps):
            # First image is skipped
            res[:, v_slice, slim(1 + i_map)] = imgs_gt[i, slch(1 + i_map)]
            res[:, v_slice, slim(1 + n_maps + i_map)] = imgs_pred[i, slch(i_map)]
    return res


class MaskedMseLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mask: torch.Tensor, y_pred: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        nmask = ~mask
        loss_obj = (y_pred[mask] - y_gt[mask]).square().mean()
        loss_back = (y_pred[nmask] - y_gt[nmask]).square().mean()
        return loss_obj + loss_back


def main(args: ArgsAaeTrain) -> int:
    args.train_root_path.mkdir(parents=True, exist_ok=True)
    tcfg = tcfg_from_args(args)
    print('Config:', args)
    print('Train Config:', tcfg)
    tcfg.create_paths()
    tcfg.to_yaml(overwrite=True)

    device = torch.device(args.device)
    print(f'Pytorch device: {device}')
    skip_cache = False
    ds = BopDataset.from_dir(tcfg.bop_root_path, tcfg.dataset_name, shuffle=True, skip_cache=skip_cache)

    objs_view = ds.get_objs_view(tcfg.obj_id, out_size=tcfg.img_size, return_tensors=True,
                                 keep_source_images=False, keep_cropped_images=False)
    ov_train, ov_val = objs_view.split((-1, 0.1))
    ov_train.set_batch_size(tcfg.train_batch_size)
    ov_val.set_batch_size(tcfg.eval_batch_size)
    ov_train.set_aug_name(AUGNAME_DEFAULT)

    n_train_batches, n_val_batches = ov_train.n_batches(), ov_val.n_batches()
    if tcfg.train_epoch_steps > 0:
        n_train_batches = tcfg.train_epoch_steps
    if tcfg.val_epoch_steps > 0:
        n_val_batches = tcfg.val_epoch_steps

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
        'lr': tcfg.learning_rate,
        'weight_decay': 0.0,
        'momentum': 0.9,
        'clip_grad': None,
        'sched': 'polynomial',
        'epochs': tcfg.epochs,
        'min_lr': 1e-6,
        'poly_power': 0.9,
        'poly_step_size': 1,
        'iter_max': tcfg.epochs * n_train_batches,
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
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = MaskedMseLoss()
    amp = False
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    if tcfg.last_checkpoint_fpath.exists():
        print(f'Loading checkpoint from {tcfg.last_checkpoint_fpath}')
        checkpoint = torch.load(tcfg.last_checkpoint_fpath)
        print(f'Checkpoint keys: {list(checkpoint.keys())}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        tcfg.last_epoch = checkpoint['last_epoch']

    epochs_left = tcfg.epochs - tcfg.last_epoch

    train_epoch_it = BopEpochIterator(
        bop_view=ov_train, n_epochs=epochs_left, n_batches_per_epoch=n_train_batches, batch_size=tcfg.train_batch_size,
        drop_last=False, shuffle_between_loops=True, aug_name=AUGNAME_DEFAULT, multiprocess=args.ds_mp_loading,
        mp_pool_size=args.ds_mp_pool_size, mp_queue_size=args.train_mp_queue_size,
        return_tensors=False, keep_source_images=False, keep_cropped_images=True,
    )
    val_epoch_it = BopEpochIterator(
        bop_view=ov_val, n_epochs=epochs_left, n_batches_per_epoch=n_val_batches, batch_size=tcfg.eval_batch_size,
        drop_last=False, shuffle_between_loops=True, multiprocess=args.ds_mp_loading,
        mp_pool_size=args.ds_mp_pool_size, mp_queue_size=args.eval_mp_queue_size,
        return_tensors=False, keep_source_images=False, keep_cropped_images=True,
    )

    tbsw = tb.SummaryWriter(log_dir=str(tcfg.train_path))
    num_updates = 0
    val_loss = None
    for epoch in range(tcfg.last_epoch + 1, tcfg.epochs + 1):
        train_it = iter(train_epoch_it.get_batch_iterator())
        pbar = trange(n_train_batches, desc=f'Epoch {epoch}', unit='batch')
        model.train()
        train_loss_win = MeanWin(5)
        for step in pbar:
            gt_item = next(train_it)
            gt_item.to_tensors()
            x = stack_imgs_maps(gt_item.maps_crop_tn, gt_item.maps_names, gt_item.imgs_crop_tn)
            y_gt = stack_imgs_maps(gt_item.maps_crop_tn, gt_item.maps_names)
            x, y_gt = x.to(device), y_gt.to(device)
            with amp_autocast():
                y_pred = model.forward(x)
                # y_pred = torch.sigmoid(y_pred)
                y_pred = (y_pred + 1) / 2
                # loss: torch.Tensor = criterion(y_pred, y_gt)
                loss: torch.Tensor = criterion(x[:, :6, ...] > 0, y_pred, y_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_updates += 1
            lr_scheduler.step_update(num_updates=num_updates)
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(f'Train. loss: {loss.item():.6f}. lr: {lr:.6f}')
            train_loss_win.add(loss.item())

            if step == 0 and epoch == 1:
                img_vis = tile_images(x, y_pred, len(gt_item.maps_names))
                tbsw.add_image('Img/Train', img_vis, epoch - 1)
                tbsw.add_scalar('Params/LearningRate', lr, epoch)
            elif step == n_train_batches - 1:
                img_vis = tile_images(x, y_pred, len(gt_item.maps_names))
                tbsw.add_image('Img/Train', img_vis, epoch)
        del train_it

        pbar.close()
        tbsw.add_scalar('Loss/Train', train_loss_win.avg(), epoch)
        lr = optimizer.param_groups[0]['lr']
        tbsw.add_scalar('Params/LearningRate', lr, epoch)
        if args.device == 'cuda':
            torch.cuda.empty_cache()

        pbar = trange(n_val_batches, desc=f'Epoch {epoch}. Val', unit='batch')
        model.eval()
        val_loss_avg = 0
        val_it = iter(val_epoch_it.get_batch_iterator())
        for step in pbar:
            gt_item = next(val_it)
            gt_item.to_tensors()
            x = stack_imgs_maps(gt_item.maps_crop_tn, gt_item.maps_names, gt_item.imgs_crop_tn)
            y_gt = stack_imgs_maps(gt_item.maps_crop_tn, gt_item.maps_names)
            x, y_gt = x.to(device), y_gt.to(device)
            with amp_autocast():
                y_pred = model.forward(x)
                # y_pred = torch.sigmoid(y_pred)
                y_pred = (y_pred + 1) / 2
                # loss: torch.Tensor = criterion(y_pred, y_gt)
                loss: torch.Tensor = criterion(x[:, :6, ...] > 0, y_pred, y_gt)
            pbar.set_postfix_str(f'Val. loss: {loss.item():.6f}')
            val_loss_avg += loss.item()

            if step == n_val_batches - 1:
                img_vis = tile_images(x, y_pred, len(gt_item.maps_names))
                tbsw.add_image('Img/Val', img_vis, epoch)
        del val_it

        pbar.close()
        val_loss_avg /= n_val_batches
        tbsw.add_scalar('Loss/Val', val_loss_avg, epoch)
        if args.device == 'cuda':
            torch.cuda.empty_cache()

        print(f'Train loss: {train_loss_win.avg():.6f}. Val loss: {val_loss_avg:.6f}'
              )
        best = False
        if val_loss is None or val_loss_avg < val_loss:
            val_loss_str = f'{val_loss}' if val_loss is None else f'{val_loss:.6f}'
            print(f'Val min loss change: {val_loss_str} --> {val_loss_avg:.6f}')
            val_loss = val_loss_avg
            best = True

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'last_epoch': epoch,
            'best_val_loss': val_loss,
        }
        print(f'Saving checkpoint to {tcfg.last_checkpoint_fpath}')
        torch.save(checkpoint, tcfg.last_checkpoint_fpath)

        if best:
            print(f'New val loss minimum: {val_loss_avg:.6f}. Saving checkpoint to {tcfg.best_checkpoint_fpath}')
            shutil.copyfile(tcfg.last_checkpoint_fpath, tcfg.best_checkpoint_fpath)

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsAaeTrain, main, 'Train Augmented Auto Encoder for single object.')


