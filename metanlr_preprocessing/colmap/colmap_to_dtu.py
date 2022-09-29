"""
Adds extra cameras.
"""

from argparse import ArgumentParser
from pathlib import Path
import copy

import cv2
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import << MUST BE HERE!!!
import open3d as o3d
import scipy.spatial as spatial
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from data_processing.components.colmap.colmap_utils import intrinsics_from_camera, extrinsics_from_image
from data_processing.components.colmap.colmap_read_write_model import read_cameras_binary, read_images_binary, qvec2rotmat
from data_processing.converters.dtu_to_pcd import load_calibration, load_K_Rt_from_P
from utils.camera_visualization import axisEqual3D
from utils import math_utils


def valid_coords(coords, resolution):
    """
    Gets validity mask for 2d image px coords.
    """
    EPS = 1e-8
    return np.logical_and(
        np.logical_and(coords[..., 0] >= 0, coords[..., 0] < resolution[0] - EPS),
        np.logical_and(coords[..., 1] >= 0, coords[..., 1] < resolution[1] - EPS))


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def main():
    """
    Load a file.
    """
    parser = ArgumentParser("Remove floaters.")
    parser.add_argument("data_path", type=Path, help="Folder with the dataset.")
    parser.add_argument("dtu_src_path", type=Path, help="Mattes.")
    parser.add_argument("dtu_stl_file", type=Path, help="Mattes.")
    parser.add_argument("--verbose", type=int, default=0, help="Mattes.")
    parser.add_argument("--factor", type=float, default=1, help="Mattes.")
    parser.add_argument("--fast_global_registration", type=int, default=0, help="Use colors?")
    parser.add_argument("--icp", type=int, default=0, help="Use colors?")
    args = parser.parse_args()

    data_path = args.data_path

    print(f'Fitting cameras for {data_path}')

    # Colmap.
    sparse_path = data_path / "sparse/0"
    colmap_cameras = read_cameras_binary(sparse_path / "cameras.bin")
    colmap_images = read_images_binary(sparse_path / "images.bin")
    colmap_view_ids = np.array([int(image.name[:6]) for image in colmap_images.values()])

    # DTU.
    dtu_img_files = sorted([x for x in (args.dtu_src_path / 'image').iterdir() if x.suffix == '.png'])
    dtu_calibs = load_calibration(args.dtu_src_path / 'cameras.npz', invert=False)

    if args.verbose:
        # This is the dense point cloud from COLMAP, default filename.
        dense_cloud_path = data_path / "dense/0/fused.ply"
        cloud = PyntCloud.from_file(str(dense_cloud_path))
        points = cloud.points[['x', 'y', 'z']].to_numpy()
        pcd_colors = cloud.points[['red', 'green', 'blue']].to_numpy()

    dtu_ex = []
    dtu_in = []
    colmap_ex = []
    colmap_in = []

    for view_id, image_file in enumerate(dtu_img_files):
        # DTU.
        dtu_calib = dtu_calibs[image_file.name]
        dtu_intrinsics, dtu_pose = load_K_Rt_from_P(dtu_calib['wm'] @ dtu_calib['sm'])
        dtu_extrinsics = np.linalg.inv(dtu_pose)
        dtu_intrinsics = dtu_intrinsics / args.factor

        if view_id not in colmap_view_ids:
            continue

        # Colmap.
        colmap_idx = np.argwhere(colmap_view_ids == view_id).flatten()[0]
        colmap_image = colmap_images[colmap_idx + 1]
        colmap_extrinsics = extrinsics_from_image(colmap_image)

        colmap_camera = colmap_cameras[colmap_image.camera_id]
        colmap_intrinsic_matrix, distortion_coeffs = intrinsics_from_camera(colmap_camera)
        colmap_intrinsic_matrix_4x4 = np.eye(4)
        colmap_intrinsic_matrix_4x4[:3, :3] = colmap_intrinsic_matrix

        def mflip(x, y, z):
            transform = np.eye(4, dtype=np.float32)
            transform[0, 0] = x
            transform[1, 1] = y
            transform[2, 2] = z
            return transform

        # v = np.concatenate([mesh.primitives[0].positions.mean(0), [1.0]])
        dtu_extrinsics = mflip(1, 1, 1) @ dtu_extrinsics @ mflip(1, 1, 1)
        # colmap_extrinsics = mflip(1, -1, -1) @ colmap_extrinsics @ mflip(1, 1, 1)
        colmap_extrinsics = mflip(1, 1, 1) @ colmap_extrinsics @ mflip(1, 1, 1)

        # Collect.
        dtu_ex += [dtu_extrinsics]
        dtu_in += [dtu_intrinsics]
        colmap_ex += [colmap_extrinsics]
        colmap_in += [colmap_intrinsic_matrix_4x4]
        continue

        if args.verbose:
            projected, _ = cv2.projectPoints(points,
                                             qvec2rotmat(colmap_image.qvec),
                                             colmap_image.tvec,
                                             colmap_intrinsic_matrix,
                                             distortion_coeffs)

            resolution = (colmap_camera.width, colmap_camera.height)
            image_points = projected[:, 0, :].astype(np.int32)
            in_this_image = valid_coords(image_points, resolution)
            image_mask = np.ones((*resolution[::-1], 3), np.float32)
            image_mask[image_points[in_this_image, 1], image_points[in_this_image, 0]] = pcd_colors[in_this_image, :]
            plt.imshow(image_mask)
            plt.show()

    # Register camera bundles.
    dtu_ex = np.stack(dtu_ex)
    dtu_in = np.stack(dtu_in)
    colmap_ex = np.stack(colmap_ex)
    colmap_in = np.stack(colmap_in)

    dtu_pose = np.linalg.inv(dtu_ex)
    colmap_pose = np.linalg.inv(colmap_ex)

    # m_rot @ COLMAP = DTU
    m_rot = dtu_pose[..., :3, :3] @ np.linalg.inv(colmap_pose[..., :3, :3])
    q_rot = [R.from_matrix(x).as_quat() for x in m_rot]
    m_rot = R.from_quat(np.mean(q_rot, 0)).as_matrix()
    m_rot4 = np.eye(4)
    m_rot4[:3, :3] = m_rot

    # m_affine @ m_rot @ colmap = DTU
    m_affine = np.eye(4)
    colmap_pos = (m_rot4 @ colmap_pose)[..., :3, 3]
    dtu_pos = dtu_pose[..., :3, 3]
    m_affine[:3, :] = cv2.estimateAffine3D(colmap_pos, dtu_pos)[1]
    m_cam_to_cam = m_affine @ m_rot4

    if args.verbose:
        print(np.array2string((m_cam_to_cam @ colmap_pose)[:, :3, 3], precision=3))
        print(np.array2string(dtu_pose[:, :3, 3], precision=3))
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca(projection="3d")
        ax.clear()
        # ax.scatter(*((dtu_pos - np.mean(dtu_pos, 0)) / np.std(dtu_pos, 0).min()).T,
        #            color=plt.get_cmap('Blues')(np.linspace(.2, 1, dtu_pos.shape[0])), label='DTU')
        # ax.scatter(*((colmap_pos - np.mean(colmap_pos, 0)) / np.std(colmap_pos, 0).min()).T,
        #            color=plt.get_cmap('Reds')(np.linspace(.2, 1, dtu_pos.shape[0])), label='COLMAP')
        ax.scatter(*(dtu_pose[..., :3, 3].T),
                   color=plt.get_cmap('Blues')(np.linspace(.2, 1, dtu_pos.shape[0])), label='DTU')
        ax.scatter(*((m_cam_to_cam @ colmap_pose)[..., :3, 3].T),
                   color=plt.get_cmap('Reds')(np.linspace(.2, 1, dtu_pos.shape[0])), label='COLMAP')
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, 0),
        )
        plt.show()

    # Load DTU.
    dtu_pcd = o3d.io.read_point_cloud(str(args.dtu_stl_file))
    print(f'Loaded DTU PCD {args.dtu_stl_file}: {dtu_pcd}.')
    dtu_scale = np.array(dtu_pcd.points).std(0).max()

    # Colmap to DTU.
    colmap_mesh_file = data_path / "dense/0/poisson.ply"
    colmap_mesh = o3d.io.read_triangle_mesh(str(colmap_mesh_file))
    print(f'Loaded COLMAP Mesh {colmap_mesh_file}: {colmap_mesh}.')
    colmap_in_colmap_world = np.array(colmap_mesh.vertices)
    colmap_in_dtu_world = math_utils.transform_points(m_cam_to_cam, colmap_in_colmap_world, True)
    colmap_in_dtu_mesh = math_utils.transform_points(dtu_calib['sm'], colmap_in_dtu_world, True)
    colmap_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(colmap_in_dtu_mesh))
    if args.verbose:
        draw_registration_result(colmap_pcd, dtu_pcd, np.eye(4))

    # Global registration.
    m_registration = np.eye(4)
    voxel_size = dtu_scale * 0.05
    if args.fast_global_registration:
        source_down, source_fpfh = preprocess_point_cloud(colmap_pcd, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(dtu_pcd, voxel_size)

        # Fast.
        print('Running FastGlobalRegistration')
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
        result_fast = o3d.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        if args.verbose:
            draw_registration_result(colmap_pcd, dtu_pcd, result_fast.transformation)

        m_registration = result_fast.transformation

        # RANSAC.
        # distance_threshold = voxel_size * 1.5
        # print(":: RANSAC registration on downsampled point clouds.")
        # print("   Since the downsampling voxel size is %.3f," % voxel_size)
        # print("   we use a liberal distance threshold %.3f." % distance_threshold)
        # result = o3d.registration.registration_ransac_based_on_feature_matching(
        #     source_down, target_down, source_fpfh, target_fpfh,
        #     distance_threshold,
        #     o3d.registration.TransformationEstimationPointToPoint(False),
        #     3, [
        #         o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(
        #             0.9),
        #         o3d.registration.CorrespondenceCheckerBasedOnDistance(
        #             distance_threshold)
        #     ], o3d.registration.RANSACConvergenceCriteria())
        # draw_registration_result(colmap_pcd, dtu_pcd, result.transformation)
        # m_registration = result.transformation

    # ICP
    if args.icp:
        print('Running ICP')
        threshold = 0.02
        trans_init = m_registration
        reg_p2p = o3d.registration.registration_icp(
            colmap_pcd, dtu_pcd, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(with_scaling=False),
            o3d.registration.ICPConvergenceCriteria(max_iteration=2000))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        if args.verbose:
            draw_registration_result(colmap_pcd, dtu_pcd, reg_p2p.transformation)
        m_registration = reg_p2p.transformation

    # Put together
    m_total = m_registration @ dtu_calib['sm'] @ m_cam_to_cam
    colmap_vertices_in_dtu = math_utils.transform_points(m_total, np.array(colmap_mesh.vertices), True)
    colmap_in_dtu_mesh = math_utils.transform_points(dtu_calib['sm'], colmap_in_dtu_world, True)
    colmap_mesh.vertices = o3d.utility.Vector3dVector(colmap_vertices_in_dtu)
    if args.verbose:
        print('Final')
        draw_registration_result(colmap_mesh, dtu_pcd, np.eye(4))

        dtu_orig = np.array(dtu_pcd.points).mean(0)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=dtu_scale, origin=[0, 0, -5 * dtu_scale] - dtu_orig)
        o3d.visualization.draw_geometries([colmap_mesh, dtu_pcd, mesh_frame], point_show_normal=False)

    # Save.
    mesh_file_out = colmap_mesh_file.parent / f'{colmap_mesh_file.stem}.dtu_space.ply'
    o3d.io.write_triangle_mesh(str(mesh_file_out), colmap_mesh)

    print('Done')


if __name__ == "__main__":
    main()
