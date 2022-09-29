"""
Tests rendered DTU mesh transformations.
"""

from pathlib import Path
import argparse
import copy

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from utils import math_utils
from utils import math_utils_torch as mut
import sdf_rendering

parser = argparse.ArgumentParser("Shows PCD")
parser.add_argument("data_path", type=Path, help="Data RGBD.")
opt = parser.parse_args()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def unproject_camera_view(camera: dict, depth_map):
    """
    Generates 3D points on the surface.
    """
    view_matrix = torch.from_numpy(camera['view_matrix'])
    projection_matrix = torch.from_numpy(camera['projection_matrix'])
    rays_o, rays_d = sdf_rendering.get_rays_all(
        resolution=torch.from_numpy(camera['resolution']).long(),
        model_matrix=torch.from_numpy(np.eye(4, dtype=np.float32)),
        view_matrix=view_matrix,
        projection_matrix=projection_matrix,
    )

    SCALE_BY_Z = True
    if SCALE_BY_Z:
        # This should be correct.
        # World->Cam
        rays_d_h = torch.cat((rays_d, torch.zeros_like(rays_d[..., :1])), -1)
        rays_d_cam = mut.transform_vectors(view_matrix, rays_d_h)
        # Scale Z by depth
        rays_d_cam_scaled = -rays_d_cam / rays_d_cam[..., 2:3] * depth_map.reshape(-1, 1)
        # Cam->World
        rays_d_scaled = mut.transform_vectors(torch.inverse(view_matrix), rays_d_cam_scaled)[..., :3]
    else:
        # This is a bit naiive. It should go real bad for wide FOV cameras.
        rays_d_scaled = rays_d * depth_map.reshape(-1, 1)

    pts = rays_o + rays_d_scaled
    return pts.reshape(depth_map.shape[-2], depth_map.shape[-1], -1)


def project_camera_view(camera: dict, pts_3d):
    """
    Projects 3D to 2D.
    """
    pts_3d_h = torch.cat((pts_3d, torch.ones_like(pts_3d[..., :1])), -1)
    pts_2d = mut.transform_vectors(torch.from_numpy(camera['projection_matrix'] @ camera['view_matrix']), pts_3d_h)
    return pts_2d[..., :2] / pts_2d[..., 3:]


def load_depth(camera: dict):
    return torch.from_numpy(np.load(opt.data_path / 'render_depth_npy' / (camera['name'] + '.npy'), allow_pickle=True))[None]


def load_rgb(camera: dict):
    return torch.from_numpy(imageio.imread(opt.data_path / 'orig_image' / camera['name']).astype(np.float32) / 255).permute(2, 0, 1)


def sample_texture(texture, pts_2d):
    # grid = pts_2d[None, ..., [1, 0]]
    grid = pts_2d[None]
    grid[..., 1] *= -1
    return F.grid_sample(texture[None], grid)[0]


def get_pcd(pts_3d, mask):
    pts_3d = pts_3d.reshape(-1, 3).numpy()
    mask = mask.reshape(-1).numpy()
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_3d[mask]))


# def scatter(ndc, values, resolution=[800, 600]):
#     ndc = ndc.reshape(-1, 2)
#     values = values.reshape(-1, values.shape[-1])

#     resolution = torch.from_numpy(np.array(resolution))
#     coords = ((ndc * 0.5 + 0.5) * resolution[None] + 0.5).long()

#     is_inside = torch.all((coords > 0) & (coords < resolution[None] - 1), dim=-1)
#     coords = coords[is_inside]
#     values = values[is_inside]

#     im = torch.zeros([resolution[1], resolution[0], values.shape[-1]])
#     im[coords[:, 1], coords[:, 0], :] = values
#     return im


def main():
    """
    Render the images.
    """
    cameras = np.load(opt.data_path / 'cameras.npy', allow_pickle=True)

    # Select views.
    src_idx = 1
    dst_idx = 34

    src_cam = cameras[src_idx]
    dst_cam = cameras[dst_idx]

    # Inputs.
    src_image = load_rgb(src_cam)
    src_depth = load_depth(src_cam)
    dst_image = load_rgb(dst_cam)
    dst_depth = load_depth(dst_cam)
    mask_src = src_depth > 1e-5
    mask_dst = dst_depth > 1e-5

    # Unproject to 3D.
    pts_src_3d = unproject_camera_view(src_cam, src_depth)
    pts_dst_3d = unproject_camera_view(dst_cam, dst_depth)

    # Vizualize 3D registration.
    draw_registration_result(get_pcd(pts_src_3d, mask_src), get_pcd(pts_dst_3d, mask_dst), np.eye(4))

    # Project dst to src.
    ndc_dst_in_src = project_camera_view(src_cam, pts_dst_3d)

    # Sample src image.
    src_image_in_dst = sample_texture(src_image, ndc_dst_in_src)

    # Show.
    fig, axs = plt.subplots(1, 3, figsize=(16, 10))
    axs[0].imshow(src_image.permute(1, 2, 0))
    axs[1].imshow(dst_image.permute(1, 2, 0))
    axs[2].imshow(src_image_in_dst.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
