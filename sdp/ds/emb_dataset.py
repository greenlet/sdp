import numpy as np
import traceback
from pathlib import Path
from typing import Optional

from sdp.utils.train import ConfigAaeTrain


class EmbDataset:
    train_cfg_fname: str = 'train_config.yaml'
    obj_ds_ids_fname: str = 'obj_ds_ids.npy'
    embs_fname: str = 'embs.npy'
    data_path: Path
    tcfg: ConfigAaeTrain
    obj_ds_ids: np.ndarray
    embs: np.ndarray

    def __init__(self, data_path: Path, tcfg: ConfigAaeTrain, obj_ds_ids: np.ndarray, embs: np.ndarray):
        self.data_path = data_path
        self.tcfg = tcfg
        self.obj_ds_ids = obj_ds_ids
        self.embs = embs

    def write_cache(self):
        print(f'EmbDataset writing cache to {self.data_path}')
        self.data_path.mkdir(exist_ok=True, parents=True)
        tcfg_fpath = self.data_path / self.train_cfg_fname
        obj_ds_ids_fpath = self.data_path / self.obj_ds_ids_fname
        embs_fpath = self.data_path / self.embs_fname
        self.tcfg.to_yaml(tcfg_fpath)
        self.obj_ds_ids.tofile(obj_ds_ids_fpath)
        self.embs.tofile(embs_fpath)

    @classmethod
    def read_cache(cls, data_path: Path) -> Optional['EmbDataset']:
        print(f'EmbDataset reading cache from {data_path}')
        tcfg_fpath = data_path / cls.train_cfg_fname
        obj_ds_ids_fpath = data_path / cls.obj_ds_ids_fname
        embs_fpath = data_path / cls.embs_fname
        not_found = False
        if not tcfg_fpath.exists():
            not_found = True
            print(f'File {tcfg_fpath} does not exist')
        if not obj_ds_ids_fpath.exists():
            not_found = True
            print(f'File {obj_ds_ids_fpath} does not exist')
        if not embs_fpath.exists():
            not_found = True
            print(f'File {embs_fpath} does not exist')
        if not_found:
            return None
        try:
            tcfg = ConfigAaeTrain.from_yaml(tcfg_fpath)
            obj_ds_ids = np.fromfile(obj_ds_ids_fpath, dtype=int)
            embs = np.fromfile(embs_fpath, dtype=np.float32)
            embs = embs.reshape((len(obj_ds_ids), -1))
            return EmbDataset(data_path, tcfg, obj_ds_ids, embs)
        except Exception as e:
            traceback.print_exception(e)
            return None

