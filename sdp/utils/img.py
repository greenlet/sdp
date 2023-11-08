from typing import Union

import cv2
import numpy as np


def make_square_img(sz: int, ch: int) -> np.ndarray:
    shape = (sz, sz) if ch == 1 else (sz, sz, ch)
    return np.zeros(shape, dtype=np.uint8)


def crop_apply_mask(imgs: Union[np.ndarray, list[np.ndarray]], mask: np.ndarray, bbox_ltwh: np.ndarray, sz_out: int,
                    bb_ext_offset: float = 0.05) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
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

