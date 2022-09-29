"""
Converts DTU dataset to our training format.
This is based on:
https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/master/project_pixels_to_world_example.py
and their DTU sample set (of 3 scans) they provided.
Probably not compatible with the original full DTU release.
"""

import argparse
from pathlib import Path
import re
from collections import OrderedDict

import numpy as np
import imageio
import open3d as o3d
import cv2

import matplotlib.pyplot as plt
import yaml
from tqdm.autonotebook import tqdm

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

import utils.math_utils as math_utils
import data_processing.components.conversions as conversions


# What view matrix we like for the center camera.
TRAGET_REFERENCE_VIEW = np.eye(4, dtype=np.float32)
TRAGET_REFERENCE_VIEW[:3, 3] = [0, 0, -1]  # Camera is in +1 and looks to negative Z
PCD_RESOLUTION = 1024
MASK_PCD = True


def compute_meta(calib, calib_is_preinverted=True):
    """
    Reprojects Kinect Depth to Kinect RGB (upsamples).
    Applies matting mask, computes normals...
    """
    if not calib_is_preinverted:
        calib = {
            'cm': np.linalg.inv(calib['cm']),
            'wm': np.linalg.inv(calib['wm']),
            'sm': np.linalg.inv(calib['sm']),
        }
    else:
        calib = {
            'cm': calib['cm'].copy(),
            'wm': calib['wm'].copy(),
            'sm': calib['sm'].copy(),
        }

    calib['cm'] = calib['cm'] @ np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)

    # Transform to world using the calibration.
    cam_to_world = calib['sm'] @ calib['wm'] @ calib['cm']

    # Create compatible GL projecton matrix.
    near = 1
    far = 10000
    gl_projection = math_utils.glFrustrum(-1, 1, -1, 1, near, far)
    # Flip the input Z because GL expects negative depth.
    z_flip = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], np.float32)
    cam_to_ndc = gl_projection @ z_flip

    ##################
    # Decompose the full world->NDC transform into MVP (= P @ V @ M).
    ##################

    # Decompose WM.
    wm_T, wm_R, wm_Z, wm_S = math_utils.decompose_matrix4x4(calib['wm'])

    # Put all together
    world_to_cam = np.linalg.inv(cam_to_world)
    mvp = cam_to_ndc @ world_to_cam
    mvp = gl_projection @ z_flip @ world_to_cam
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['sm'] @ calib['wm'] @ calib['cm'])
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['sm'] @ calib['wm'] @ calib['cm'])
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(calib['wm']) @ np.linalg.inv(calib['sm'])
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(wm_T @ wm_R @
                                                                              wm_Z @ wm_S) @ np.linalg.inv(calib['sm'])
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(wm_Z @
                                                                              wm_S) @ np.linalg.inv(wm_T @ wm_R) @ np.linalg.inv(calib['sm'])
    # Split to MVP.
    projection_matrix = gl_projection @ z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(wm_Z @ wm_S)
    view_matrix = np.linalg.inv(wm_T @ wm_R)
    model_matrix = np.linalg.inv(calib['sm'])

    ##################
    # Cleanup the projection - move all sign flips to the view matrix.
    ##################

    # Split the core GL projection and the rest
    pre_proj = z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(wm_Z @ wm_S)
    projection_matrix = gl_projection @ pre_proj

    # Decompose "the rest" in inverse order.
    pre_T, pre_R, pre_Z, pre_S = math_utils.decompose_matrix4x4(np.linalg.inv(pre_proj))
    pre_T = np.linalg.inv(pre_T)
    pre_R = np.linalg.inv(pre_R)
    pre_Z = np.linalg.inv(pre_Z)
    pre_S = np.linalg.inv(pre_S)
    pre_proj2 = pre_S @ pre_Z @ pre_R @ pre_T
    assert np.max(np.abs(pre_proj2 - pre_proj)) < 1e-3
    projection_matrix = gl_projection @ pre_S @ pre_Z @ pre_R @ pre_T

    # Make the scaling component of projection only positive (move the negativity to view matrix).
    pre_Z_sign = np.sign(pre_Z)
    projection_matrix = gl_projection @ pre_S @ pre_Z @ pre_Z_sign @ np.linalg.inv(pre_Z_sign) @ pre_R @ pre_T

    # Move rotation and translation to view matrix.
    projection_matrix = gl_projection @ pre_S @ pre_Z @ pre_Z_sign
    view_matrix = np.linalg.inv(pre_Z_sign) @ pre_R @ pre_T @ view_matrix

    # Validate projection matrix.
    proj_params = math_utils.decompose_projection_matrix(projection_matrix)
    assert np.abs(proj_params['f'] - far) / far < 1e-2
    assert np.abs(proj_params['n'] - near) / near < 1e-2
    assert proj_params['l'] < proj_params['r']
    assert proj_params['b'] < proj_params['t']

    # Flip view matrix XZ to make it compatible with our coordinate system.
    view_flip = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], np.float32)
    view_matrix = view_matrix @ view_flip
    model_matrix = np.linalg.inv(view_flip) @ model_matrix
    assert np.abs(np.linalg.det(view_matrix) - 1) < 1e-3

    # Inverse projection would be cm^-1 @ wm^-1 @ sm^-1 @ points_world
    meta = {
        'view': view_matrix,
        'projection': projection_matrix,
        'm_pcd_to_original': np.linalg.inv(calib['sm']) @ np.linalg.inv(model_matrix),
    }
    return meta


def process_depth_frame(img_file, mask_file, depth_file, calib, args):
    """
    Reprojects Kinect Depth to Kinect RGB (upsamples).
    Applies matting mask, computes normals...
    """
    meta_reference = compute_meta(calib)

    # Load.
    im = imageio.imread(img_file).astype(np.float32) / 255
    mask = imageio.imread(mask_file)[..., 0].astype(np.float32) / 255
    resolution = np.array([im.shape[1], im.shape[0]], int)
    if depth_file is not None:
        try:
            depth = imageio.imread(depth_file)
        except:
            imageio.plugins.freeimage.download()
            depth = imageio.imread(depth_file)
    else:
        depth = None

    if depth is not None:
        # 2D coordinates.
        depth_size = np.array([depth.shape[1], depth.shape[0]], int)
        np.testing.assert_array_equal(resolution, depth_size)
        coords = np.stack(np.meshgrid(
            np.linspace(-1, 1, depth_size[0]),
            np.linspace(-1, 1, depth_size[1]),
        ), -1)
        coords = np.stack(np.meshgrid(
            (np.arange(depth_size[0]).astype(np.float32)) / (depth_size[0] - 1) * 2 - 1,
            (np.arange(depth_size[1]).astype(np.float32)) / (depth_size[1] - 1) * 2 - 1,
        ), -1)

    # We index images as numpy does - top to bottom
    if depth is not None:
        coords[..., 1] *= -1
    calib['cm'] = calib['cm'] @ np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)

    # Linearize.
    if depth is not None:
        coords = coords.reshape(-1, 2).astype(np.float32)
        depth = depth.reshape(-1)
    colors = im[..., :3].reshape(-1, 3)
    mask = (mask > 0.5).reshape(-1)

    # Apply depth mask.
    if depth is not None:
        mask_depth = np.logical_not(np.isinf(depth))
        mask_full = np.logical_and(mask, mask_depth)
        coords = coords[mask_full, :]
        depth = depth[mask_full]
    else:
        mask_full = mask
    colors = colors[mask_full, :]

    if depth is not None:
        # Project coords to 3D.
        points = np.concatenate((coords, np.ones_like(coords)), -1)
        points[..., :3] *= depth[..., None].repeat(3, axis=1)

    # Transform to world using the calibration.
    cam_to_world = calib['sm'] @ calib['wm'] @ calib['cm']
    if depth is not None:
        points_world = math_utils.transform_points(cam_to_world, points)
        points_world = points_world[..., :3]

    # Convert back.
    world_to_cam = np.linalg.inv(cam_to_world)
    if depth is not None:
        points2 = math_utils.transform_points(world_to_cam, points_world, return_euclidean=True)
        assert np.max(np.abs(points[..., :3] - points2)) < 1e-3
        points_ndc = points[:, :3] / points[:, 2:3].repeat(3, 1)
        assert np.max(np.abs(points_ndc[..., :2] - coords)) < 1e-3

    # Create compatible GL projecton matrix.
    near = 1
    far = 10000
    gl_projection = math_utils.glFrustrum(-1, 1, -1, 1, near, far)
    # Flip the input Z because GL expects negative depth.
    z_flip = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], np.float32)
    cam_to_ndc = gl_projection @ z_flip
    if depth is not None:
        points_ndc_2 = math_utils.transform_points(cam_to_ndc, points2, return_euclidean=True)
        assert np.max(np.abs(points_ndc[:, :2] - points_ndc_2[:, :2])) < 1e-5

        # Check Depth.
        gl_depth = points_ndc_2[..., 2]
        #clip_space_depth = 2.0 * gl_depth - 1.0
        linear_depth = 2.0 * near * far / (far + near - gl_depth * (far - near))
        assert np.max(np.abs(linear_depth - depth) / depth) < 1e-3

    # Chain together.
    world_to_ndc = cam_to_ndc @ world_to_cam
    if depth is not None:
        points_ndc_3 = math_utils.transform_points(world_to_ndc, points_world, return_euclidean=True)
        assert np.max(np.abs(points_ndc[:, :2] - points_ndc_3[:, :2])) < 1e-5

    ##################
    # Decompose the full world->NDC transform into MVP (= P @ V @ M).
    ##################

    # Decompose WM.
    wm_T, wm_R, wm_Z, wm_S = math_utils.decompose_matrix4x4(calib['wm'])

    # Put all together
    mvp = cam_to_ndc @ world_to_cam
    mvp = gl_projection @ z_flip @ world_to_cam
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['sm'] @ calib['wm'] @ calib['cm'])
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['sm'] @ calib['wm'] @ calib['cm'])
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(calib['wm']) @ np.linalg.inv(calib['sm'])
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(wm_T @ wm_R @
                                                                              wm_Z @ wm_S) @ np.linalg.inv(calib['sm'])
    mvp = gl_projection @ z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(wm_Z @
                                                                              wm_S) @ np.linalg.inv(wm_T @ wm_R) @ np.linalg.inv(calib['sm'])
    # Split to MVP.
    projection_matrix = gl_projection @ z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(wm_Z @ wm_S)
    view_matrix = np.linalg.inv(wm_T @ wm_R)
    model_matrix = np.linalg.inv(calib['sm'])

    ##################
    # Cleanup the projection - move all sign flips to the view matrix.
    ##################

    # Split the core GL projection and the rest
    pre_proj = z_flip @ np.linalg.inv(calib['cm']) @ np.linalg.inv(wm_Z @ wm_S)
    projection_matrix = gl_projection @ pre_proj

    # Decompose "the rest" in inverse order.
    pre_T, pre_R, pre_Z, pre_S = math_utils.decompose_matrix4x4(np.linalg.inv(pre_proj))
    pre_T = np.linalg.inv(pre_T)
    pre_R = np.linalg.inv(pre_R)
    pre_Z = np.linalg.inv(pre_Z)
    pre_S = np.linalg.inv(pre_S)
    pre_proj2 = pre_S @ pre_Z @ pre_R @ pre_T
    assert np.max(np.abs(pre_proj2 - pre_proj)) < 1e-3
    projection_matrix = gl_projection @ pre_S @ pre_Z @ pre_R @ pre_T

    # Make the scaling component of projection only positive (move the negativity to view matrix).
    pre_Z_sign = np.sign(pre_Z)
    projection_matrix = gl_projection @ pre_S @ pre_Z @ pre_Z_sign @ np.linalg.inv(pre_Z_sign) @ pre_R @ pre_T

    # Move rotation and translation to view matrix.
    projection_matrix = gl_projection @ pre_S @ pre_Z @ pre_Z_sign
    view_matrix = np.linalg.inv(pre_Z_sign) @ pre_R @ pre_T @ view_matrix

    if depth is not None:
        # Validate.
        mvp = projection_matrix @ view_matrix @ model_matrix
        points_ndc_4 = math_utils.transform_points(mvp, points_world, return_euclidean=True)
        assert np.max(np.abs(points_ndc[:, :2] - points_ndc_4[:, :2])) < 1e-5

    # Flip projection matrix
    # projection_flip = np.array([
    #     [-1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ], np.float32)
    #projection_matrix = projection_matrix @ projection_flip
    #view_matrix = np.linalg.inv(projection_flip) @ view_matrix

    # Validate projection matrix.
    proj_params = math_utils.decompose_projection_matrix(projection_matrix)
    assert np.abs(proj_params['f'] - far) / far < 1e-2
    assert np.abs(proj_params['n'] - near) / near < 1e-2
    assert proj_params['l'] < proj_params['r']
    assert proj_params['b'] < proj_params['t']

    # Flip view matrix XZ to make it compatible with our coordinate system.
    view_flip = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], np.float32)
    view_matrix = view_matrix @ view_flip
    model_matrix = np.linalg.inv(view_flip) @ model_matrix
    assert np.abs(np.linalg.det(view_matrix) - 1) < 1e-3

    # # Flip model matrix Y.
    # model_flip = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ], np.float32)
    # model_matrix = model_flip @ model_matrix

    if depth is not None:
        # Transform model
        points_world = math_utils.transform_points(model_matrix, points_world, return_euclidean=True)

    # Inverse projection would be cm^-1 @ wm^-1 @ sm^-1 @ points_world
    meta = {
        'name': f'{img_file.stem}',
        'view': view_matrix,
        'projection': projection_matrix,
        'resolution': resolution,
        'is_in_world_coords': True,
        'image_file': img_file.name,
        'depth_file': depth_file.name if depth_file is not None else "",
        'mask_file': mask_file.name,
        'm_pcd_to_original': np.linalg.inv(calib['sm']) @ np.linalg.inv(model_matrix),
    }

    # Make sure consistent.
    for k, v in meta_reference.items():
        assert np.max(np.abs(v - meta[k])) < 1e-8

    pcd_world = None
    if depth is not None:
        pcd_world = o3d.geometry.PointCloud()
        pcd_world.points = o3d.utility.Vector3dVector(points_world)
        pcd_world.colors = o3d.utility.Vector3dVector(colors)

        # Validate projection
        if args.visualize:
            conversions.validate_pcd_projection(im, meta, pcd_world, meta, args.output_path)

        # Remove PCD outliers.
        cl, ind = pcd_world.remove_statistical_outlier(nb_neighbors=20,
                                                       std_ratio=2.0)
        # display_inlier_outlier(pcd_world, ind)
        pcd_world = cl
        points_world = np.array(pcd_world.points)

        # Estimate normals.
        pcd_world = conversions.estimate_normals(pcd_world, view_matrix)

        if args.visualize:
            scale = points_world.std(0).max()
            orig = points_world.mean(0)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=scale, origin=[0, 0, -5 * scale] - orig)
            o3d.visualization.draw_geometries([pcd_world, mesh_frame], point_show_normal=False)

    return pcd_world, meta


def export_meta(pcd, meta, input_path: Path, output_path: Path):
    """
    Defered HDD write allows for global centering.
    """
    # Load extra inputs.
    im = imageio.imread(input_path / 'image' / meta['image_file'])
    mask = imageio.imread(input_path / 'mask' / meta['mask_file'])
    depth = None
    if meta['depth_file']:
        depth = imageio.imread(input_path / 'depth' / meta['depth_file'])

    # Write out.
    output_base_file = str(output_path / f'{meta["name"]}')

    # Write RGB.
    output_file = output_base_file + '_rgb.png'
    imageio.imwrite(output_file, im)

    # Write Mask.
    output_file = output_base_file + '_mask.png'
    imageio.imwrite(output_file, mask)

    # Write RGB's meta.
    output_file = output_base_file + '_rgb_meta.npy'
    np.save(output_file, meta)

    if depth is not None:
        # Write GT depth for viz purposes.
        output_file = output_base_file + '_depth.png'
        depth1d = depth.reshape(-1)
        is_inf = np.isinf(depth1d)
        depth1d[np.isinf(depth1d)] = 0
        depth = depth1d.reshape(depth.shape)
        d_min = depth1d[np.logical_not(is_inf)].min()
        d_max = depth1d[np.logical_not(is_inf)].max()
        viz_depth = (depth - d_min) / (d_max - d_min)
        conversions.imwritef(output_file, viz_depth)

    if False and pcd is not None:
        # Render normals into image.
        normals = np.array(pcd.normals) * 0.5 + 0.5
        normals[np.any(np.isnan(normals), 1), :] = 0
        pcd_normals = o3d.geometry.PointCloud()
        pcd_normals.points = pcd.points
        pcd_normals.colors = o3d.utility.Vector3dVector(normals)
        im_normals = conversions.render_pcd(pcd_normals, meta['view'], meta['projection'], meta['resolution'])[0]

        # Write GT normals for viz purposes.
        output_file = output_base_file + '_normal.png'
        conversions.imwritef(output_file, im_normals)


def load_calibration(calibration_file: Path, invert=True):
    """
    Loads calibration_file.
    """
    cam = np.load(calibration_file)
    img_dir = calibration_file.parent / 'image'
    img_files = sorted([x for x in img_dir.iterdir() if x.suffix == '.png'])

    calibs = OrderedDict()
    for i, image_file in enumerate(img_files):
        # Apply inversion here.
        cm = cam.get('camera_mat_%d' % i).astype(np.float32)
        wm = cam.get('world_mat_%d' % i).astype(np.float32)
        sm = cam.get('scale_mat_%d' % i).astype(np.float32)
        if invert:
            cm = np.linalg.inv(cm)
            wm = np.linalg.inv(wm)
            sm = np.linalg.inv(sm)
        calibs[image_file.name] = {
            'cm': cm,
            'wm': wm,
            'sm': sm,
        }

    return calibs


def compute_scene_stats(pcd, metas):
    """
    Computes statistics.
    """
    stats = {
        'pcd_center': np.array([0, 0, 0], np.float32),
        'pcd_radius': 0.0,
        'camera_intersection': np.array([0, 0, 0], np.float32),
        'camera_radius': 0.0,
    }
    if pcd is not None:
        points = np.array(pcd.points)
        coord_min = np.min(points, axis=0)
        coord_max = np.max(points, axis=0)
        pcd_center = (coord_min + coord_max) / 2
        pcd_radius = np.linalg.norm(points - pcd_center, axis=1).max()
        stats['pcd_center'] = pcd_center
        stats['pcd_radius'] = pcd_radius

    # Compute camera stats.
    cam2worlds = [np.linalg.inv(m['view']) for m in metas]
    cam_positions = [cam2world[:3, 3] for cam2world in cam2worlds]
    cam_view_directions = [-cam2world[:3, 2] for cam2world in cam2worlds]
    cam_intersection = conversions.find_lines_intersection_3D(np.array(cam_positions), np.array(cam_view_directions))
    cam_radius = np.linalg.norm(cam_positions - cam_intersection, axis=1).min()
    stats['camera_intersection'] = cam_intersection
    stats['camera_radius'] = cam_radius

    return stats


def load_K_Rt_from_P(P):
    P = P[:3, :4]

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def render_pcd(pcd, resolution, sm, wm, cm):
    """
    Renders PCD under projection.
    """
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)

    # Project points.

    # Custom:
    world_to_cam = cm @ wm @ sm
    pts_cam = math_utils.transform_points(cm @ wm, points, return_euclidean=True)
    pts_screen = pts_cam[..., :2] / pts_cam[..., 2:3]
    coords_2d = ((pts_screen + 1) / 2 * resolution).astype(int)

    # Based on IDR code:
    P = wm @ sm
    intrinsics, pose = load_K_Rt_from_P(P)
    points_unscaled = math_utils.transform_points(np.linalg.inv(sm), points, return_euclidean=True)
    pose_inv = np.linalg.inv(pose)
    pts_screen = cv2.projectPoints(points_unscaled, cv2.Rodrigues(pose_inv[:3, :3])[
                                   0], pose_inv[:3, 3], intrinsics[:3, :3], distCoeffs=None)[0]
    coords_2d_idr = pts_screen[:, 0, :].astype(int)
    # Compare to validate.
    assert np.max(np.abs(coords_2d - coords_2d_idr)) <= 1

    # Remove out of canvas.
    mask = conversions.valid_coords(coords_2d, resolution)
    coords_2d_masked = coords_2d[mask, :]
    colors = np.array(pcd.colors)[mask, :]

    # Paint.
    canvas = np.zeros((resolution[1], resolution[0], 3), np.float32)
    canvas[coords_2d_masked[:, 1], coords_2d_masked[:, 0], :] = colors
    return canvas, coords_2d


def mask_pointcloud(pcd: o3d.geometry.PointCloud, dataset_path: Path, visualize=False) -> o3d.geometry.PointCloud:
    """
    Masks the pointcloud.
    """
    print(f'Masking PCD using masks in {dataset_path}...')
    calibs = load_calibration(dataset_path / 'cameras.npz', invert=False)
    img_dir = dataset_path / 'image'
    img_files = sorted([x for x in img_dir.iterdir() if x.suffix == '.png'])
    mask_dir = dataset_path / 'mask'
    mask_files = sorted([x for x in mask_dir.iterdir() if x.suffix == '.png'])

    pts_invalid = np.zeros((len(pcd.points),), dtype=bool)

    # Process 3D images.
    for i, (img_file, mask_file) in enumerate(zip(img_files, mask_files)):
        calib = calibs[img_file.name]
        render, coords_2d = render_pcd(pcd, [1600, 1200], calib['sm'], calib['wm'], calib['cm'])

        # Load mask.
        mask = imageio.imread(mask_file)[..., 0].astype(np.float32) / 255
        is_invalid = 1 - mask
        is_invalid_samples = conversions.resample_image(is_invalid, coords_2d) > 0.5
        pts_invalid = np.logical_or(pts_invalid, is_invalid_samples)
        print(f'[{i}/{len(img_files)}] {(1-np.mean(is_invalid_samples)) * 100:.2f}% valid points => {(1-np.mean(pts_invalid)) * 100:.2f}% total')

        if visualize:
            # Load GT img.
            im_gt = imageio.imread(img_file).astype(np.float32) / 255

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(im_gt)
            axs[1].imshow(render)
            plt.show()

    # Mask the points.
    pts = np.array(pcd.points)
    normals = np.array(pcd.normals)
    colors = np.array(pcd.colors)

    pcd_masked = o3d.geometry.PointCloud()
    pcd_masked.points = o3d.utility.Vector3dVector(pts[~pts_invalid, :])
    pcd_masked.normals = o3d.utility.Vector3dVector(normals[~pts_invalid, :])
    pcd_masked.colors = o3d.utility.Vector3dVector(colors[~pts_invalid, :])
    return pcd_masked


def get_transform_dtu_to_ours(dataset_path) -> o3d.geometry.PointCloud:
    """
    Get matrix that transform from DTU/IDR/DVR coordinates to ours.
    """
    # Test Points.
    pts_idr = np.random.rand(28, 3)

    # Load both DVR and IDR calirbations.
    # They unfortunatelly differ in wm and we used DVR in some scenes (65,106,118).
    calibs_idr = load_calibration(dataset_path / 'cameras.npz')
    calibs_dvr = None
    if (dataset_path / 'cameras_dvr.npz').is_file():
        calibs_dvr = load_calibration(dataset_path / 'cameras_dvr.npz')
    if calibs_dvr is None:
        calibs_dvr = calibs_idr
    calib_idr = calibs_idr[list(calibs_idr.keys())[0]]
    calib_dvr = calibs_dvr[list(calibs_dvr.keys())[0]]

    # 4) Project to camera using IDR calibration.
    pts_cam = math_utils.transform_points(
        calib_idr['cm'] @ calib_idr['wm'], pts_idr, return_euclidean=True)

    # 3) UnProject from camera using DVR calibration.
    world_to_cam_dvr = calib_dvr['cm'] @ calib_dvr['wm'] @ calib_dvr['sm']
    pts_dvr = math_utils.transform_points(np.linalg.inv(world_to_cam_dvr), pts_cam, return_euclidean=True)

    # 2) Apply the model scaling.
    metas = [compute_meta(c, calib_is_preinverted=True) for c in calibs_dvr.values()]
    model_matrix = metas[0]['m_pcd_to_original'] @ calib_dvr['sm']
    pts_pcd_raw = math_utils.transform_points(model_matrix, pts_dvr, return_euclidean=True)

    # 1) Apply the model centering.
    m_transform = conversions.normalize_pcd_poses([(None, m) for m in metas], TRAGET_REFERENCE_VIEW)
    pts_pcd_raw = math_utils.transform_points(np.linalg.inv(m_transform), pts_pcd_raw, return_euclidean=True)

    # Combine all together.
    m_full = np.linalg.inv(
        m_transform) @ model_matrix @ np.linalg.inv(world_to_cam_dvr) @ calib_idr['cm'] @ calib_idr['wm']

    pts_pcd_raw2 = math_utils.transform_points(m_transform, pts_idr, return_euclidean=True)
    # assert pts_pcd_raw == pts_pcd_raw2

    return m_full


def transform_pcd(pcd: o3d.geometry.PointCloud, m_transform: np.array) -> o3d.geometry.PointCloud:
    """
    Converts to our coords.
    """

    # Transform points and normals.
    pcd_pts = math_utils.transform_points(m_transform, np.array(pcd.points), return_euclidean=True)
    pcd_normals = math_utils.transform_normals(m_transform, np.array(pcd.normals))

    # Write out.
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(pcd_pts)
    pcd_new.normals = o3d.utility.Vector3dVector(pcd_normals)
    pcd_new.colors = pcd.colors
    return pcd_new


def process_scan(input_path: Path, output_path: Path, stl_filename: Path, opts):
    """
    Processes single DTU scan.
    """
    # Prepare output.
    output_path.mkdir(0o777, True, True)

    # Load calibrations.
    calibs = load_calibration(input_path / 'cameras.npz')

    # Scan all frames.
    img_dir = input_path / 'image'
    img_files = sorted([x for x in img_dir.iterdir() if x.suffix == '.png'])
    mask_dir = input_path / 'mask'
    mask_files = sorted([x for x in mask_dir.iterdir() if x.suffix == '.png'])
    depth_dir = input_path / 'depth'
    if depth_dir.is_dir():
        depth_files = sorted([x for x in depth_dir.iterdir() if x.suffix in ['.exr']])
    else:
        depth_files = [None] * len(img_files)

    ###################################################
    # Process RGBD images. Depth is optional.
    pcds_with_metas = []
    for i, (img, mask, depth) in enumerate(zip(img_files, mask_files, depth_files)):
        print(f'Processing image {i} of {len(img_files)}.')
        calib = calibs[img.name]
        pcds_with_metas += [process_depth_frame(img, mask, depth, calib, opts)]
        # if len(pcds) == 2:
        #     break

    # Enforce desired reference view matrix. This finalizes OUR coordinates.
    conversions.normalize_pcd_poses(pcds_with_metas, TRAGET_REFERENCE_VIEW)
    #####################################################

    # Get total transform. Contains everything including the normalize_pcd_poses.
    # Trust me, I am an engineer.
    m_dtu_to_ours = get_transform_dtu_to_ours(input_path)

    pcd = None
    pcds = [x[0] for x in pcds_with_metas if x[0] is not None]
    metas = [x[1] for x in pcds_with_metas if x[1] is not None]
    if stl_filename is not None:
        # Prefer external PCD (already merged)
        dtu_pcd = o3d.io.read_point_cloud(str(stl_filename))
        if MASK_PCD:
            # Mask out points by manual masks.
            dtu_pcd = mask_pointcloud(dtu_pcd, input_path, opts.visualize)
            # Transform DTU to Our coordinates.
        pcd = transform_pcd(dtu_pcd, m_dtu_to_ours)
    elif pcds:
        # Use PCDs built from RGBDs (Already in our coordinates).
        # Merge them.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate([x.points for x in pcds], 0))
        pcd.normals = o3d.utility.Vector3dVector(np.concatenate([x.normals for x in pcds], 0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate([x.colors for x in pcds], 0))
        # Remove PCD outliers.
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                std_ratio=2.0)

    if pcd is not None:
        # Downsample PCD.
        if PCD_RESOLUTION > 0:
            points = np.array(pcd.points)
            bbox = points.max(0) - points.min(0)
            d_volume = bbox.max() / PCD_RESOLUTION
            pcd = pcd.voxel_down_sample(d_volume)
            print(f'Downsampled PCD from {points.shape[0]} to {len(pcd.points)} points.')

        # Save as PLY.
        output_file = output_path / "merged_pcd.ply"
        print(f'Exporting point cloud to {output_file}...')
        o3d.io.write_point_cloud(str(output_file), pcd)

        # Save meta.
        np.save(output_path / "merged_pcd_meta.npy", {
            'm_pcd_to_original': metas[0]['m_pcd_to_original'],
            'm_dtu_to_ours': m_dtu_to_ours,  # Raw DTU @ THIS = Our PCD
        })

    # Compute scene statistics.
    stats = compute_scene_stats(pcd, metas)
    output_file = output_path / f"scene_stats.npy"
    np.save(output_file, stats)

    # Validate results.
    for i, meta_a in enumerate(metas):
        if pcd is not None:
            # Reproject all images.
            if opts.render_all:
                im = imageio.imread(img_files[i]).astype(np.float32) / 255
                for meta_b in metas:
                    conversions.validate_pcd_projection(im, meta_a, pcd, meta_b, args.output_path)

    # Save
    for i, meta in enumerate(metas):
        export_meta(pcd, meta, input_path, output_path)

    if pcd is not None and opts.visualize:
        # Visualize.
        pcd.normals = o3d.utility.Vector3dVector()

        scale = np.array(pcd.points).std(0).max()
        orig = np.array(pcd.points).mean(0)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale, origin=[0, 0, -5 * scale] - orig)
        o3d.visualization.draw_geometries([pcd, mesh_frame], point_show_normal=False)


def main():
    parser = argparse.ArgumentParser("DTU dataset to XYZ converter.")
    parser.add_argument("input_path", type=Path, help="Folder with array data.")
    parser.add_argument("output_path", type=Path, help="Output folder.")
    parser.add_argument('--stl_points', type=Path, help="STL Points file.")
    parser.add_argument('-v', '--visualize', action="store_true", default=False,
                        help='Interactive mode with visualizations.')
    parser.add_argument('--render_all', action="store_true", default=False,
                        help='Render all projections.')
    args = parser.parse_args()

    # Single file case.
    if (args.input_path / 'cameras.npz').is_file():
        print('Processing single input.')
        process_scan(args.input_path, args.output_path, args.stl_points, args)
        return

    input_folders = sorted([x for x in args.input_path.iterdir() if x.is_dir() and (x / 'cameras.npz').is_file()])
    with tqdm(total=len(input_folders)) as pbar:
        for i, input_dir in enumerate(input_folders):
            output_path = args.output_path / input_dir.name
            scan_id = int(re.match(r'[^\d]*(\d+)[^\d]*', input_dir.name).group(1))
            stl_path = args.stl_points / f'stl{scan_id:03d}_total.ply'
            tqdm.write(f'[{i}/{len(input_folders)}] Processing {input_dir}...')
            process_scan(input_dir, output_path, stl_path, args)
            pbar.update(1)

    print('DONE')


if __name__ == "__main__":
    main()
