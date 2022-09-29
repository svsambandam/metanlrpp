"""
Render a ply file in the DTU space.
"""

from pathlib import Path
import argparse
import yaml
import os
from argparse import Namespace
import shutil
import sys
if sys.platform.startswith("linux"):
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # nopep8

import trimesh
import imageio
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data_processing.converters.dtu_to_pcd import load_calibration, load_K_Rt_from_P
from data_processing.components.conversions import intrinsics_to_gl_frustrum, gl_frustrum_to_intrinsics
from utils import math_utils
import data_processing.datasets.dataio_sdf as dataio_sdf
import pyrender


def get_dataset(dataset_path, load_images=True):
    args = Namespace()
    args.load_pcd = False
    args.load_images = load_images
    args.load_im_scale = 1.0
    args.fit_sphere = False
    args.fit_plane = False
    args.randomize_cameras = False
    args.device = 'cpu'
    args.dataset_type = 'sinesdf_static'
    args.test_views = ''
    args.is_test_only = True
    args.scene_radius_scale = 1.0
    args.work_radius = 0.99
    args.im_scale = 1.0
    args.opt_sdf_direct = False
    args.scene_normalization = 'cache,yaml,pcd,camera'
    return dataio_sdf.DatasetSDF(Path(dataset_path), args)


def mflip(x, y, z):
    transform = np.eye(4, dtype=np.float32)
    transform[0, 0] = x
    transform[1, 1] = y
    transform[2, 2] = z
    return transform


def main():
    """
    Render the images.
    """
    parser = argparse.ArgumentParser("Shows PCD")
    parser.add_argument("mesh_file", type=Path, help="Mesh PLY.")
    parser.add_argument("reference_data_path", type=Path, help="Either IDR DTU path or NLR dataset")
    parser.add_argument("dataset_type", type=str, help="Select which dataset format to use [dtu|nlr]")
    parser.add_argument("--colors", type=int, default=1, help="Use colors?")
    parser.add_argument("--factor", type=int, default=2, help="Scale resolution (down)?")
    opt = parser.parse_args()

    background_color = [1.0, 1.0, 1.0]
    if opt.colors:
        scene = pyrender.Scene(ambient_light=[0, 0, 0], bg_color=background_color)
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=background_color)

    # Mesh
    print(f'Loading {opt.mesh_file}...')
    mesh = pyrender.Mesh.from_trimesh(trimesh.load_mesh(opt.mesh_file), smooth=True)
    if not opt.colors:
        mesh.primitives[0].material = pyrender.MetallicRoughnessMaterial(
            # alphaMode='BLEND',
            #emissiveFactor=(0, 0, 0),
            metallicFactor=0,
            # roughnessFactor=0.1,
            #baseColorFactor=(1, 1, 1, 1)
        )
        mesh.primitives[0].color_0 = None  # mesh.primitives[0].color_0 * 0 + 1

    # Create a PyRender camera corresponding to each Camera extrinsics.
    model_matrix = []
    views = []
    views_export = []
    resolution = np.array([1600, 1200]) // opt.factor

    if opt.dataset_type.lower() == 'nlr':
        # Render in the PCD=NLR space.
        print(f'Render in the PCD = NLR space.')
        nlr_dataset = get_dataset(opt.reference_data_path, False)
        meta = np.load(opt.reference_data_path / 'merged_pcd_meta.npy', allow_pickle=True)
        m_dtu_to_nlr = meta.reshape(1)[0]['m_dtu_to_ours']

        #model_matrix = nlr_dataset.model_matrix.cpu().numpy() @ m_dtu_to_nlr
        model_matrix = m_dtu_to_nlr
        for i, view in enumerate(nlr_dataset.image_views):
            m_view = view.view_matrix.detach().cpu().numpy()

            intrinsics = gl_frustrum_to_intrinsics(view.projection_matrix.detach().cpu().numpy(), resolution)
            pyrender_camera = pyrender.IntrinsicsCamera(intrinsics[0, 0], intrinsics[1, 1],
                                                        intrinsics[0, 2] - 0.5, intrinsics[1, 2] - 0.5,
                                                        zfar=10000)

            image_name = f'{i:06d}.png'
            views += [{
                'name': image_name,
                'camera': pyrender_camera,
                'resolution': resolution,
                'view_matrix': m_view
            }]

            views_export += [{
                'name': image_name,
                'resolution': resolution,
                'projection_matrix': intrinsics_to_gl_frustrum(intrinsics[:3, :3], resolution),
                'view_matrix': m_view,
                'model_matrix': model_matrix,
            }]
    else:
        # Render in the DTU space.
        print(f'Render in the original DTU dataset space.')
        dtu_calibs = load_calibration(opt.reference_data_path / 'cameras.npz', invert=False)

        for image_name, dtu_calib in dtu_calibs.items():
            P = dtu_calib['wm'] @ dtu_calib['sm']
            intrinsics, pose = load_K_Rt_from_P(P)
            intrinsics[0, 0] /= opt.factor
            intrinsics[1, 1] /= opt.factor
            intrinsics[0, 2] /= opt.factor
            intrinsics[1, 2] /= opt.factor
            pyrender_camera = pyrender.IntrinsicsCamera(intrinsics[0, 0], intrinsics[1, 1],
                                                        intrinsics[0, 2] - 0.5, intrinsics[1, 2] - 0.5,
                                                        zfar=10000)

            extrinsics = np.linalg.inv(pose)
            m_view = mflip(1, -1, -1) @ extrinsics @ mflip(1, 1, 1)

            model_matrix = np.linalg.inv(dtu_calib['sm'])

            views += [{
                'name': image_name,
                'camera': pyrender_camera,
                'resolution': resolution,
                'view_matrix': m_view
            }]

            views_export += [{
                'name': image_name,
                'resolution': resolution,
                'projection_matrix': intrinsics_to_gl_frustrum(intrinsics[:3, :3], resolution),
                'view_matrix': m_view,
                'model_matrix': model_matrix,
            }]
    scene.add(mesh, pose=model_matrix)

    out_path_root = opt.mesh_file.parent / "rgbd_export_test"
    if opt.colors:
        output_path_rgb = out_path_root / "render_rgb"
    else:
        output_path_rgb = out_path_root / "render_shaded"
    output_path_depth_npy = out_path_root / 'render_depth_npy'
    output_path_depth = out_path_root / 'render_depth'

    output_path_rgb.mkdir(0o777, True, True)
    output_path_depth_npy.mkdir(0o777, True, True)
    output_path_depth.mkdir(0o777, True, True)

    if not opt.colors:
        # Lights.
        # Front.
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=550)
        light_pose = np.eye(4)
        light_pose[:3, :3] = R.from_euler('xy', [-45 + 180, 10], degrees=True).as_matrix()
        scene.add(light, pose=light_pose)

        # Rear.
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=120)
        light_pose = np.eye(4)
        light_pose[:3, :3] = R.from_euler('xy', [0 + 180, -10], degrees=True).as_matrix()
        scene.add(light, pose=light_pose)

    # Render.
    for i, view in enumerate(views):

        camera_node = scene.add(view['camera'], pose=np.linalg.inv(view['view_matrix']))

        resolution = np.array(view['resolution'])
        renderer = pyrender.OffscreenRenderer(*resolution)
        flags = pyrender.constants.RenderFlags.FLAT if opt.colors else pyrender.constants.RenderFlags.NONE
        im_out, depth = renderer.render(scene, flags)
        print(f'[{i}/{len(views)}] Mean color = {im_out.mean((0, 1))} | Mean depth = {depth.mean((0, 1))}')

        scene.remove_node(camera_node)

        # Save.
        rgb_file = output_path_rgb / f"{view['name']}"
        print(f'Saving {rgb_file}...')
        imageio.imwrite(rgb_file, im_out)
        depth_file = output_path_depth / f"{view['name']}"
        imageio.imwrite(depth_file, (depth[..., None] - np.min(depth)) / (np.max(depth) - np.min(depth)))
        depth_file = output_path_depth_npy / f"{view['name']}.npy"
        np.save(depth_file, depth)

    # Export cameras.
    out_filename = out_path_root / 'cameras.npy'
    print(f'Writing calibration to {out_filename}...')
    np.save(out_filename, views_export)

    if opt.dataset_type.lower() == 'dtu':
        # Copy over the original textures and masks.
        output_path_image = out_path_root / 'orig_image'
        output_path_mask = out_path_root / 'orig_mask'
        output_path_image.mkdir(0o777, True, True)
        output_path_mask.mkdir(0o777, True, True)
        images = sorted([x for x in (opt.dtu_data_path / 'image').iterdir() if x.suffix == '.png'])
        masks = sorted([x for x in (opt.dtu_data_path / 'mask').iterdir() if x.suffix == '.png'])
        for i, view in enumerate(views):
            shutil.copy(images[i], output_path_image / view['name'])
            shutil.copy(masks[i], output_path_mask / view['name'])
        shutil.copy(opt.mesh_file, out_path_root / opt.mesh_file.name)

    print('DONE')


if __name__ == "__main__":
    main()
