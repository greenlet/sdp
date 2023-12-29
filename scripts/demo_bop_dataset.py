from datetime import datetime
import os
from matplotlib import pyplot as plt
from pathlib import Path
import sys

import numpy as np

from sdp.ds.bop_data import BopModelsInfo, read_models_info, id_to_str, read_meshes
from sdp.ds.bop_dataset import BopDataset, AUGNAME_DEFAULT
from sdp.utils.img import crop_apply_mask

DATA_PATH = Path(os.path.expandvars('$HOME/data'))
BOP_PATH = DATA_PATH / 'bop'
ITODD_SUBDIR = 'itodd'
ITODD_BOP_PATH = BOP_PATH / ITODD_SUBDIR
print(f'BOP path: {BOP_PATH}')
TRAIN_ROOT_PATH = DATA_PATH / 'train_aae'
ITODD_MODELS_PATH = ITODD_BOP_PATH / 'models'


def demo_01_mp():
    skip_cache = False
    ds = BopDataset.from_dir(BOP_PATH, 'itodd', shuffle=True, skip_cache=skip_cache)
    multiprocess = True
    aug_name = AUGNAME_DEFAULT
    # aug_name = None
    # return_tensors, keep_source_mages, keep_cropped_images = False, True, True
    return_tensors, keep_source_mages, keep_cropped_images = True, False, False
    ov = ds.get_objs_view(1, out_size=256, aug_name=aug_name)
    n_batches = 1000
    batch_size = 50
    mp_queue_size = 5
    mp_pool_size = 7
    ds.set_mp_pool_size(mp_pool_size)
    batch_it = ov.get_batch_iterator(
        n_batches, batch_size=batch_size, multiprocess=multiprocess, mp_queue_size=mp_queue_size, aug_name=aug_name,
        return_tensors=return_tensors, keep_source_images=keep_source_mages, keep_cropped_images=keep_cropped_images)

    t1 = datetime.now()
    n = 50
    for i in range(n):
        b = next(batch_it)
        print(i, len(b.imgs), len(b.imgs_crop_tn))
    t2 = datetime.now()
    print(f'n = {n}. Avg time: {(t2 - t1).total_seconds() / n:.03} sec.')


if __name__ == '__main__':
    demo_01_mp()


