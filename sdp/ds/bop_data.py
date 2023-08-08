import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdp.utils.data import read_json


@dataclass
class BopSymmetryContinuous:
    axis: np.ndarray
    offset: np.ndarray


@dataclass
class BopModelsInfo:
    id: int
    diameter: float
    pt_min: np.ndarray
    pt_max: np.ndarray
    size: np.ndarray
    syms_discrete: list[np.ndarray]
    syms_continuous: list[BopSymmetryContinuous]

    @staticmethod
    def from_json(id: int, data: dict[str, Any]) -> 'BopModelsInfo':
        diameter = data['diameter']
        pt_min = np.array([data['min_x'], data['min_y'], data['min_z']], dtype=float)
        pt_max = np.array([data['max_x'], data['max_y'], data['max_z']], dtype=float)
        size = np.array([data['size_x'], data['size_y'], data['size_z']], dtype=float)
        syms_discrete = []
        for sd in data.get('symmetries_discrete', []):
            sd = np.array(sd, dtype=float).reshape((4, 4))
            syms_discrete.append(sd)
        syms_discrete = syms_discrete
        syms_continuous = []
        for sc in data.get('symmetries_continuous', []):
            axis = np.array(sc['axis'], dtype=float)
            offset = np.array(sc['offset'], dtype=float)
            sc = BopSymmetryContinuous(axis, offset)
            syms_continuous.append(sc)
        syms_continuous = syms_continuous
        return BopModelsInfo(
            id=id, diameter=diameter, pt_min=pt_min, pt_max=pt_max, size=size,
            syms_discrete=syms_discrete, syms_continuous=syms_continuous,
        )


def read_models_info(models_info_fpath: Path) -> dict[int, BopModelsInfo]:
    data = read_json(models_info_fpath)
    return {int(k): BopModelsInfo.from_json(k, v) for k, v in data.items()}


@dataclass
class BopSceneCameraEntry:
    scene_id: int
    K: np.ndarray
    depth_scale: float

    @staticmethod
    def from_json(scene_id: int, data: dict[str, Any]) -> 'BopSceneCameraEntry':
        K = np.array(data['cam_K'], dtype=float).reshape((3, 3))
        depth_scale = data['depth_scale']
        return BopSceneCameraEntry(scene_id=scene_id, K=K, depth_scale=depth_scale)


BopSceneCamera = dict[int, BopSceneCameraEntry]


def read_scene_camera(scene_camera_fpath: Path) -> BopSceneCamera:
    data = read_json(scene_camera_fpath)
    return {int(k): BopSceneCameraEntry.from_json(int(k), v) for k, v in data.items()}


@dataclass
class BopObjGt:
    scene_id: int
    img_id: int
    obj_id: int
    R_m2c: np.ndarray
    t_m2c: np.ndarray

    @staticmethod
    def from_json(scene_id: int, img_id: int, data: dict[str, Any]) -> 'BopObjGt':
        obj_id = int(data['obj_id'])
        R_m2c = np.array(data['cam_R_m2c'], dtype=float)
        t_m2c = np.array(data['cam_t_m2c'], dtype=float)
        return BopObjGt(
            scene_id=scene_id, img_id=img_id, obj_id=obj_id, R_m2c=R_m2c, t_m2c=t_m2c,
        )


BopImgGt = list[BopObjGt]
BopSceneGt = dict[int, BopImgGt]


def read_scene_gt(scene_gt_fpath: Path) -> BopSceneGt:
    scene_gt_json = read_json(scene_gt_fpath)
    scene_id = int(scene_gt_fpath.parent.name)
    scene_gt = {}
    for img_id, img_gt_json in scene_gt_json.items():
        img_id = int(img_id)
        img_gt = []
        for i, obj_gt_json in enumerate(img_gt_json):
            obj_gt = BopObjGt.from_json(scene_id, img_id, obj_gt_json)
            img_gt.append(obj_gt)
        scene_gt[img_id] = img_gt
    return scene_gt


@dataclass
class BopSceneObjGtInfo:
    scene_id: int
    bbox_obj: np.ndarray
    bbox_visib: np.ndarray
    px_count_all: int
    px_count_valid: int
    px_count_visib: int
    visib_fract: float

    @staticmethod
    def from_json(scene_id: int, data: dict[str, Any]) -> 'BopSceneObjGtInfo':
        bbox_obj = np.array(data['bbox_obj'], dtype=float)
        bbox_visib = np.array(data['bbox_visib'], dtype=float)
        px_count_all = data['px_count_all']
        px_count_valid = data['px_count_valid']
        px_count_visib = data['px_count_visib']
        visib_fract = float(data['visib_fract'])
        return BopSceneObjGtInfo(
            scene_id=scene_id, bbox_obj=bbox_obj, bbox_visib=bbox_visib, px_count_all=px_count_all,
            px_count_valid=px_count_valid, px_count_visib=px_count_visib, visib_fract=visib_fract,
        )


BopSceneGtInfo = dict[int, list[BopSceneObjGtInfo]]


def read_scene_gt_info(scene_gt_info_fpath: Path) -> BopSceneGtInfo:
    scene_gt_info_json = read_json(scene_gt_info_fpath)

    scene_gt_info = {}
    for scene_id, scene_objs_gt_info_json in scene_gt_info_json.items():
        scene_id = int(scene_id)
        scene_objs_gt_info = []
        for i, obj_gt_info_json in enumerate(scene_objs_gt_info_json):
            obj_gt_info = BopSceneObjGtInfo.from_json(scene_id, obj_gt_info_json)
            scene_objs_gt_info.append(obj_gt_info)
        scene_gt_info[scene_id] = scene_objs_gt_info
    return scene_gt_info


