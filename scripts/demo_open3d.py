import copy
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

    vis = o3d.visualization.Visualizer()
    vis.add_geometry(m3d)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    tr = np.eye(4)
    tr[:3, :3] = R_m2c
    tr[:3, 3] = t_m2c
    m3d.transform(tr)
    renderer.scene.add_geometry(f'obj_{id_to_str(obj_id)}', m3d, None)
    renderer.scene.add_camera('cam', cam_int)
    renderer.scene.set_active_camera('cam')

    ctr = renderer.scene.get_view_control()
    cam_param = ctr.convert_to_pinhole_camera_parameters()
    # cam_param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    cam_param.intrinsic = cam_int
    cam_param.extrinsic = cam_ext
    ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)

    img_o3d = renderer.render_to_image()
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
    models_path = ITODD_MODELS_PATH
    models_info = read_models_info(models_path / 'models_info.json')
    model_info = models_info[obj_id]
    print(model_info)
    meshes = read_meshes(ITODD_MODELS_PATH, obj_id)
    mesh = meshes[obj_id]
    m3d_src = mesh.mesh_o3d
    m3d_src.compute_vertex_normals()
    m3d_src.paint_uniform_color((0.5, 0, 0))
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
        m3d = copy.deepcopy(m3d_src)
        m3d.transform(tr)
        if vis is None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=width, height=height, left=200, top=200)
        vis.clear_geometries()
        vis.add_geometry(m3d)
        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        # cam_param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
        cam_param.intrinsic = cam_int
        cam_param.extrinsic = cam_ext
        ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(f'data/image_{i:05d}.png')

        img_o3d = vis.capture_screen_float_buffer(False)
        img = np.asarray(img_o3d)
        print(img.shape, img.dtype, img.min(), img.max())
        img = (img * 255).astype(np.uint8)
        bb = ro['bbox_visib_ltwh']
        # Workaround for cx, cy bug (https://github.com/isl-org/Open3D/issues/6016)
        bb[:2] += (width / 2 - cx), (height / 2 - cy)
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
        # bb[:2] = (width / 2 - cx), (height / 2 - cy)
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


def fix_normals(mesh: o3d.geometry.TriangleMesh, vdir: np.ndarray, vert: bool):
    if vert:
        normals = np.asarray(mesh.vertex_normals)
    else:
        normals = np.asarray(mesh.triangle_normals)
    vdir = vdir / np.linalg.norm(vdir)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    prod = normals @ vdir.reshape((3, 1))
    mul = np.ones((normals.shape[0], 1))
    mul[prod < 0] = -1
    normals = normals * mul
    if vert:
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    else:
        mesh.triangle_normals = o3d.utility.Vector3dVector(normals)


def demo_06_headless():
    obj_id = 1
    out_size = 512
    ds = BopDataset.from_dir(BOP_PATH, ITODD_SUBDIR)
    models_path = ITODD_BOP_PATH / 'models'
    # models_path = ITODD_BOP_PATH / 'models_eval'
    models_info = read_models_info(models_path / 'models_info.json')
    model_info = models_info[obj_id]
    print(model_info)
    meshes = read_meshes(models_path, obj_id)
    mesh = meshes[obj_id]
    m3d_src = copy.deepcopy(mesh.mesh_o3d)
    m3d_src.compute_vertex_normals()
    # m3d_src.compute_triangle_normals()
    wf_src: o3d.geometry.LineSet = o3d.geometry.LineSet.create_from_triangle_mesh(m3d_src)

    # m3d_src.paint_uniform_color((0.5, 0, 0))
    objs_view = ds.get_objs_view(obj_id, out_size=out_size)
    renderer = None
    obj_key = f'obj_{id_to_str(obj_id)}'

    for i in range(100):
        gt = objs_view.get_gt_imgs_masks(i)
        ri, ro = gt.df_img.iloc[0], gt.df_obj.iloc[0]
        width, height = ri['img_width'], ri['img_height']

        K = ri['cam_K']
        cx, cy = K[0, 0], K[1, 1]
        t_m2c, R_m2c = ro['t_m2c'] * mesh.mul_to_meters, ro['R_m2c']

        tr = np.eye(4)
        tr[:3, :3] = R_m2c
        tr[:3, 3] = t_m2c
        m3d = copy.deepcopy(m3d_src)
        m3d.transform(tr)
        print(t_m2c)
        # fix_normals(m3d, -t_m2c, True)
        # fix_normals(m3d, np.array([0, 0, -1]), True)
        # fix_normals(m3d, -t_m2c, False)
        obj = m3d
        # wf = copy.deepcopy(wf_src)
        # wf.transform(tr)
        # obj = wf

        # Define a simple unlit Material.
        # (The base color does not replace the arrows' own colors.)
        mtl = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
        mtl.base_color = [1.0, 0., 0., 0.5]  # RGBA
        # mtl.base_metallic = 0.7
        # mtl.base_roughness = 0.3
        mtl.shader = "defaultLit"

        if renderer is None:
            renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
            renderer.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                             75000)
            renderer.scene.scene.enable_sun_light(True)
            # renderer.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
            # renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
        renderer.scene.remove_geometry(obj_key)
        renderer.scene.add_geometry(obj_key, obj, mtl)
        # else:
        #     renderer.scene.scene.update_geometry(obj_key, m3d.vertices, 1)

        # if renderer is None:
        #     renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        #     renderer.scene.add_geometry(obj_key, m3d, mtl)
        #     # renderer.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
        #     renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
        # # else:
        #     renderer.scene.add_geometry(obj_key, m3d, mtl)

        near_plane = 0.1
        far_plane = 50.0
        # set_projection(self: open3d.cuda.pybind.visualization.rendering.Camera, intrinsics: numpy.ndarray[
        #     numpy.float64[3, 3]], near_plane: float, far_plane: float, image_width: float, image_height: float) -> None
        renderer.scene.camera.set_projection(K, near_plane, far_plane, width, height)

        # look_at(self: open3d.cuda.pybind.visualization.rendering.Camera, center: numpy.ndarray[numpy.float32[3, 1]], eye:
        # numpy.ndarray[numpy.float32[3, 1]], up: numpy.ndarray[numpy.float32[3, 1]]) -> None

        # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        center = [0, 0, 1]  # look_at target
        eye = [0, 0, 0]  # camera position
        up = [0, -1, 0]  # camera orientation
        renderer.scene.camera.look_at(center, eye, up)

        img_o3d = renderer.render_to_image()

        # plt.imshow(img_o3d)

        img = np.asarray(img_o3d)
        # print(img.min(), img.mean(), img.max(), img.dtype, img.shape)

        # fig, axes = plt.subplots(1, 2)
        # fig.set_size_inches(18.5, 10.5)
        # axes[0].imshow(img)
        # axes[1].imshow(gt.imgs[0])
        # plt.show()

        bb = ro['bbox_visib_ltwh']
        # Workaround for cx, cy bug (https://github.com/isl-org/Open3D/issues/6016)
        # bb[:2] = (width / 2 - cx), (height / 2 - cy)
        imgs_crop, mask_out, _, _ = crop_apply_mask(img, gt.masks[0] > 0, bb, out_size)
        # plt.imshow(img)
        img_crop = imgs_crop[0]

        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(18.5, 10.5)
        axes[0, 0].imshow(gt.imgs_crop[0])
        axes[0, 1].imshow(img_crop)
        axes[1, 0].imshow(gt.maps_crop['noc'][0])
        axes[1, 1].imshow(gt.maps_crop['norm'][0])

        plt.show()


if __name__ == '__main__':
    # demo_01_simple()
    # demo_02_show_bop_obj()
    # demo_03_headless()
    demo_04_bop_image()
    # demo_05_bop_images()
    # demo_06_headless()

