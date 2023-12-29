import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_yaml import to_yaml_str, parse_yaml_file_as
from typing import Optional


class ArgsAaeBase(BaseModel):
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
    eval_batch_size: Optional[int] = Field(
        None,
        required=False,
        description='Inference batch size (number of images for single inference validation/evaluation network run).',
        cli=('--eval-batch-size',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run training on. Can have values: "cpu", "gpu"',
        cli=('--device',)
    )
    ds_mp_loading: bool = Field(
        False,
        required=False,
        description='Enables multiprocessing for dataset loader.',
        cli=('--ds-mp-loading',)
    )
    ds_mp_pool_size: Optional[int] = Field(
        None,
        required=False,
        description='Number of processes in multiprocess pool when DS_MP_LOADING = true. Default value is the total '
                    'number of cores on the machine.',
        cli=('--ds-mp-pool-size',)
    )
    eval_mp_queue_size: int = Field(
        5,
        required=False,
        description='Eval dataset multiprocess data loader queue size.',
        cli=('--eval-mp-queue-size',)
    )


class MeanWin:
    _window_size: int
    _vals: list[float]
    _full: bool = False
    _i: int = 0

    def __init__(self, window_size: int):
        assert window_size > 0
        self._window_size = window_size
        self._vals = [0.0] * self._window_size

    def add(self, val: float):
        self._vals[self._i] = val
        self._i += 1
        if self._i == len(self._vals):
            self._full = True
            self._i = 0

    def avg(self) -> float:
        if self._full:
            return sum(self._vals) / len(self._vals)
        n = self._i + 1
        return sum(self._vals[:n]) / n


class ConfigAaeTrain(BaseModel):
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
    train_epoch_steps: int
    val_epoch_steps: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
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
    def from_yaml(cls, fpath: Optional[Path] = None) -> 'ConfigAaeTrain':
        if fpath.is_dir():
            fpath = fpath / cls._fname
        return parse_yaml_file_as(ConfigAaeTrain, fpath)

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


def img_to_tensor(img: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
    img = img.astype(np.float32) / 255
    if mask is not None:
        # u = np.unique(mask)
        # assert len(u) == 2 and u[0] == 0 and u[1] > 0
        img[mask == 0] = 0
    res: torch.Tensor = torch.from_numpy(img)
    return res


def imgs_list_to_tensors(imgs: list[np.ndarray], masks: Optional[list[np.ndarray]] = None) -> list[torch.Tensor]:
    res = []
    for i, img in enumerate(imgs):
        mask = None if masks is None else masks[i]
        res.append(img_to_tensor(img, mask))
    return res


def imgs_dict_to_tensors(imgs_dict: dict[str, list[np.ndarray]], masks: Optional[list[np.ndarray]] = None) -> dict[str, list[torch.Tensor]]:
    return {k: imgs_list_to_tensors(imgs, masks) for k, imgs in imgs_dict.items()}

