import time

from copy import deepcopy

import os
from matplotlib import pyplot as plt
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

from sdp.ds.bop_data import BopModelsInfo, read_models_info, id_to_str, read_meshes
from sdp.ds.bop_dataset import BopDataset
from sdp.utils.img import crop_apply_mask

DATA_PATH = Path(os.path.expandvars('$HOME/data'))
BOP_PATH = DATA_PATH / 'bop'
ITODD_SUBDIR = 'itodd'
ITODD_BOP_PATH = BOP_PATH / ITODD_SUBDIR
print(f'BOP path: {BOP_PATH}')
TRAIN_ROOT_PATH = DATA_PATH / 'train_aae'
ITODD_MODELS_PATH = ITODD_BOP_PATH / 'models'


def demo_01_simple():
    fpath = ITODD_BOP_PATH / 'models' / 'obj_000001.ply'
    pcd = o3d.io.read_point_cloud(str(fpath))
    mesh = o3d.io.read_triangle_mesh(str(fpath))
    o3d.visualization.draw_geometries([mesh, pcd])


def demo_02_show_bop_obj():
    obj_id = 1
    out_size = 256
    ds = BopDataset.from_dir(BOP_PATH, ITODD_SUBDIR)
    models_info = read_models_info(ITODD_MODELS_PATH / 'models_info.json')
    model_info = models_info[obj_id]
    print(model_info)
    meshes = read_meshes(ITODD_MODELS_PATH, obj_id)
    mesh = meshes[obj_id]
    m3d = mesh.mesh_o3d
    print(np.asarray(m3d.vertices))
    m3d.compute_vertex_normals()
    m3d.paint_uniform_color((0.5, 0, 0))
    objs_view = ds.get_objs_view(obj_id, out_size=out_size)
    gt = objs_view.get_gt_imgs_masks(0)
    ri, ro = gt.df_img.iloc[0], gt.df_obj.iloc[0]
    width, height = ri['img_width'], ri['img_height']
    K = ri['cam_K']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    t_m2c, R_m2c = ro['t_m2c'] * mesh.mul_to_meters, ro['R_m2c']
    cam_int = o3d.camera.PinholeCameraIntrinsic(width, height, K)
    cam_ext = np.eye(4)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, left=200, top=200)
    # m3d.rotate(ro['R_m2c'])
    # m3d.translate(ro['t_m2c'])
    tr = np.eye(4)
    tr[:3, :3] = R_m2c
    tr[:3, 3] = t_m2c
    print(f'Transform:\n{tr}')
    m3d.transform(tr)
    vis.add_geometry(m3d)
    ctr = vis.get_view_control()
    cam_param = ctr.convert_to_pinhole_camera_parameters()
    # cam_param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    cam_param.intrinsic = cam_int
    cam_param.extrinsic = cam_ext
    ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)
    vis.run()
    vis.destroy_window()


def demo_03_headless():
    obj_id = 1
    out_size = 256
    ds = BopDataset.from_dir(BOP_PATH, ITODD_SUBDIR)
    models_info = read_models_info(ITODD_MODELS_PATH / 'models_info.json')
    model_info = models_info[obj_id]
    print(model_info)
    meshes = read_meshes(ITODD_MODELS_PATH, obj_id)
    mesh = meshes[obj_id]
    m3d = mesh.mesh_o3d
    print(np.asarray(m3d.vertices))
    m3d.compute_vertex_normals()
    m3d.paint_uniform_color((0.5, 0, 0))
    objs_view = ds.get_objs_view(obj_id, out_size=out_size)
    gt = objs_view.get_gt_imgs_masks(0)
    ri, ro = gt.df_img.iloc[0], gt.df_obj.iloc[0]
    width, height = ri['img_width'], ri['img_height']
    K = ri['cam_K']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    t_m2c, R_m2c = ro['t_m2c'] * mesh.mul_to_meters, ro['R_m2c']
    cam_int = o3d.camera.PinholeCameraIntrinsic(width, height, K)
    cam_ext = np.eye(4)

    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    tr = np.eye(4)
    tr[:3, :3] = R_m2c
    tr[:3, 3] = t_m2c
    m3d.transform(tr)
    render.scene.add_geometry(f'obj_{id_to_str(obj_id)}', m3d)
    render.scene.add_camera('cam', cam_int)
    render.scene.set_active_camera('cam')

    ctr = render.scene.get_view_control()
    cam_param = ctr.convert_to_pinhole_camera_parameters()
    # cam_param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    cam_param.intrinsic = cam_int
    cam_param.extrinsic = cam_ext
    ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)

    img_o3d = render.render_to_image()
    plt.imshow(img_o3d)
    # ctr = vis.get_view_control()
    # cam_param = ctr.convert_to_pinhole_camera_parameters()
    # # cam_param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    # cam_param.intrinsic = cam_int
    # cam_param.extrinsic = cam_ext
    # ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)
    # vis.run()
    # vis.destroy_window()


def demo_04_bop_image():
    obj_id = 1
    out_size = 256
    ds = BopDataset.from_dir(BOP_PATH, ITODD_SUBDIR)
    models_info = read_models_info(ITODD_MODELS_PATH / 'models_info.json')
    model_info = models_info[obj_id]
    print(model_info)
    meshes = read_meshes(ITODD_MODELS_PATH, obj_id)
    mesh = meshes[obj_id]
    m3d = mesh.mesh_o3d
    print(np.asarray(m3d.vertices))
    m3d.compute_vertex_normals()
    m3d.paint_uniform_color((0.5, 0, 0))
    objs_view = ds.get_objs_view(obj_id, out_size=out_size)
    vis = None
    for i in range(10):
        gt = objs_view.get_gt_imgs_masks(i)
        ri, ro = gt.df_img.iloc[0], gt.df_obj.iloc[0]
        width, height = ri['img_width'], ri['img_height']
        K = ri['cam_K']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        t_m2c, R_m2c = ro['t_m2c'] * mesh.mul_to_meters, ro['R_m2c']
        cam_int = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        cam_int.intrinsic_matrix = K
        cam_ext = np.eye(4)
        tr = np.eye(4)
        tr[:3, :3] = R_m2c
        tr[:3, 3] = t_m2c
        m3d.transform(tr)
        if vis is None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=width, height=height, left=200, top=200)
            vis.add_geometry(m3d)
        else:
            vis.update_geometry(m3d)
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        # cam_param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
        cam_param.intrinsic = cam_int
        cam_param.extrinsic = cam_ext
        ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(f'data/image_{i:05d}.png')

        # img_o3d = vis.capture_screen_float_buffer(False)
        # print(img_o3d)
        # img = np.asarray(img_o3d)
        # img = (img * 255).astype(np.uint8)
        # bb = ro['bbox_visib_ltwh']
        # # Workaround for cx, cy bug (https://github.com/isl-org/Open3D/issues/6016)
        # bb[:2] = (width / 2 - cx), (height / 2 - cy)
        # imgs_crop, mask_out, _, _ = crop_apply_mask(img, gt.masks[0] > 0, bb, out_size)
        # # plt.imshow(img)
        # img_crop = imgs_crop[0]

        # fig, axes = plt.subplots(2, 2)
        # print(img_crop.shape, img_crop.dtype)
        # axes[0, 0].imshow(gt.imgs_crop[0])
        # axes[0, 1].imshow(img_crop)
        # axes[1, 0].imshow(gt.maps_crop['noc'][0])
        # axes[1, 1].imshow(gt.maps_crop['norm'][0])
        # plt.show()

        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # time.sleep(1)
    vis.destroy_window()


def demo_05_bop_images():
    obj_id = 1
    out_size = 256
    ds = BopDataset.from_dir(BOP_PATH, ITODD_SUBDIR)
    models_info = read_models_info(ITODD_MODELS_PATH / 'models_info.json')
    model_info = models_info[obj_id]
    print(model_info)
    meshes = read_meshes(ITODD_MODELS_PATH, obj_id)
    mesh = meshes[obj_id]
    m3d = mesh.mesh_o3d
    print(np.asarray(m3d.vertices))
    m3d.compute_vertex_normals()
    m3d.paint_uniform_color((0.5, 0, 0))
    objs_view = ds.get_objs_view(obj_id, out_size=out_size)
    for i in range(10):
        gt = objs_view.get_gt_imgs_masks(i)
        ri, ro = gt.df_img.iloc[0], gt.df_obj.iloc[0]
        width, height = ri['img_width'], ri['img_height']
        K = ri['cam_K']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        t_m2c, R_m2c = ro['t_m2c'] * mesh.mul_to_meters, ro['R_m2c']
        cam_int = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        cam_int.intrinsic_matrix = K
        cam_ext = np.eye(4)
        tr = np.eye(4)
        tr[:3, :3] = R_m2c
        tr[:3, 3] = t_m2c
        m3d.transform(tr)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, left=200, top=200)
        vis.add_geometry(m3d)
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        # cam_param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
        cam_param.intrinsic = cam_int
        cam_param.extrinsic = cam_ext
        ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)

        # vis.poll_events()
        # vis.update_renderer()

        # vis.capture_screen_image(f'data/image_{i:05d}.png')

        img_o3d = vis.capture_screen_float_buffer(True)
        print(img_o3d)
        img = np.asarray(img_o3d)
        img = (img * 255).astype(np.uint8)
        bb = ro['bbox_visib_ltwh']
        # Workaround for cx, cy bug (https://github.com/isl-org/Open3D/issues/6016)
        bb[:2] = (width / 2 - cx), (height / 2 - cy)
        imgs_crop, mask_out, _, _ = crop_apply_mask(img, gt.masks[0] > 0, bb, out_size)
        # plt.imshow(img)
        img_crop = imgs_crop[0]

        fig, axes = plt.subplots(2, 2)
        print(img_crop.shape, img_crop.dtype)
        axes[0, 0].imshow(gt.imgs_crop[0])
        axes[0, 1].imshow(img_crop)
        axes[1, 0].imshow(gt.maps_crop['noc'][0])
        axes[1, 1].imshow(gt.maps_crop['norm'][0])
        plt.show()

        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # time.sleep(1)
        # vis.destroy_window()
        # vis.close()


if __name__ == '__main__':
    # demo_01_simple()
    # demo_02_show_bop_obj()
    # demo_03_headless()
    # demo_04_bop_image()
    demo_05_bop_images()

