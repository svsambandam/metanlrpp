"""
Utilities for parsing colmap data.
"""
import numpy as np

import data_processing.components.colmap.colmap_read_write_model as colmap_reader


def inspect_camera(cam):
    """
    Prints the camera parameters.
    """
    print("id: ", cam.id)
    print("model: ", cam.model)
    print("width: ", cam.width)
    print("height: ", cam.height)
    print("params: ", cam.params)


def inspect_image(img):
    """
    Prints some image properties.
    """
    print("id: ", img.id)
    print("qvec: ", img.qvec)
    print("tvec: ", img.tvec)
    print("camera_id: ", img.camera_id)
    print("name: ", img.name)
    # print("xys: ", img.xys)
    # print("point3D_ids: ", img.point3D_ids)


def inspect_point(pt):
    """
    Inspect the point.
    """
    print("id: ", pt.id)
    print("xyz: ", pt.xyz)
    print("rgb: ", pt.rgb)
    print("error: ", pt.error)
    print("image_ids: ", pt.image_ids)
    print("point2D_idxs: ", pt.point2D_idxs)


def intrinsics_from_camera(camera):
    """
    Returns the camera matrix and distortion coefficients.
    """
    def params_to_camera_mat_opencv(params):
        """
        Create the camera matrix from a list of fx, fy, cx, cy.
        """
        assert len(params) == 4
        return np.array([(params[0], 0, params[2]), (0, params[1], params[3]), (0, 0, 1)])

    def params_to_camera_mat_simple_radial(params):
        """
        Create the camera matrix from a list of fx, fy, cx, cy.
        """
        assert len(params) == 3
        return np.array([(params[0], 0, params[1]), (0, params[0], params[2]), (0, 0, 1)])

    if camera.model in ('OPENCV', 'FULL_OPENCV', 'PINHOLE'):
        # Get intrinsic matrix.
        camera_matrix = params_to_camera_mat_opencv(camera.params[:4])
        # Remaining parameters are distortion.
        dist_coeff = camera.params[4:]  # TODO Verify this works if camera params only length 4.
    elif camera.model in ('SIMPLE_RADIAL', 'SIMPLE_PINHOLE'):
        camera_matrix = params_to_camera_mat_simple_radial(camera.params[:3])
        dist_coeff = np.zeros(4)
        if camera.model == 'SIMPLE_RADIAL':
            dist_coeff[0] = camera.params[3]
    else:
        raise NotImplementedError(f"Camera model {camera.model} is not supported at this time.")

    return camera_matrix, dist_coeff


def extrinsics_from_image(img):
    """
    Given an image, extract the extrinsic matrix (4x4).
    """
    # Get rotation matrix.
    R = colmap_reader.qvec2rotmat(img.qvec)
    # Translation vector.
    t = img.tvec
    # Stitch to 4x4 matrix.
    world2cam = np.eye(4)
    world2cam[:3, :3] = R
    world2cam[:3, 3] = t

    return world2cam
