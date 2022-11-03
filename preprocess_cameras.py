import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import argparse
from glob import glob
import os

import os
from glob import glob
# import torch

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(np.split(np.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = np.take(model_input['uv'], indx, axis=1)
        data['object_mask'] = np.take(model_input['object_mask'], indx, axis=1)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = np.concatenate([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = np.concatenate([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs


def get_Ps(cameras,number_of_cameras):
    Ps = []
    for i in range(0, number_of_cameras):
        P = cameras['world_mat_%d' % i][:3, :].astype(np.float64)
        Ps.append(P)
    return np.array(Ps)


#Gets the fundamental matrix that transforms points from the image of camera 2, to a line in the image of
#camera 1
def get_fundamental_matrix(P_1,P_2):
    P_2_center=np.linalg.svd(P_2)[-1][-1, :]
    epipole=P_1@P_2_center
    epipole_cross=np.zeros((3,3))
    epipole_cross[0,1]=-epipole[2]
    epipole_cross[1, 0] = epipole[2]

    epipole_cross[0,2]=epipole[1]
    epipole_cross[2, 0] = -epipole[1]

    epipole_cross[1, 2] = -epipole[0]
    epipole_cross[2, 1] = epipole[0]

    F = epipole_cross@P_1 @ np.linalg.pinv(P_2)
    return F



# Given a point (curx,cury) in image 0, get the  maximum and minimum
# possible depth of the point, considering the second image silhouette (index j)
def get_min_max_d(curx, cury, P_j, silhouette_j, P_0, Fj0, j):
    # transfer point to line using the fundamental matrix:
    cur_l_1=Fj0 @ np.array([curx,cury,1.0]).astype(np.float)
    cur_l_1 = cur_l_1 / np.linalg.norm(cur_l_1[:2])

    # Distances of the silhouette points from the epipolar line:
    dists = np.abs(silhouette_j.T @ cur_l_1)
    relevant_matching_points_1 = silhouette_j[:, dists < 0.7]

    if relevant_matching_points_1.shape[1]==0:
        return (0.0,0.0)
    X = cv2.triangulatePoints(P_0, P_j, np.tile(np.array([curx, cury]).astype(np.float),
                                                (relevant_matching_points_1.shape[1], 1)).T,
                              relevant_matching_points_1[:2, :])
    depths = P_0[2] @ (X / X[3])
    reldepth=depths >= 0
    depths=depths[reldepth]
    if depths.shape[0] == 0:
        return (0.0, 0.0)

    min_depth=depths.min()
    max_depth = depths.max()

    return min_depth,max_depth

#get all fundamental matrices that trasform points from camera 0 to lines in Ps
def get_fundamental_matrices(P_0, Ps):
    Fs=[]
    for i in range(0,Ps.shape[0]):
        F_i0 = get_fundamental_matrix(Ps[i],P_0)
        Fs.append(F_i0)
    return np.array(Fs)

def get_all_mask_points(masks_dir):
    mask_paths = sorted(glob_imgs(masks_dir))
    mask_points_all=[]
    mask_ims = []
    for path in mask_paths:
        img = mpimg.imread(path)
        # print('curmask', img.shape)
        if len(img.shape) > 2:
            cur_mask = img.max(axis=2) > 0.5
            mask_points = np.where(img.max(axis=2) > 0.5)
        else:
            cur_mask = img > 0.5
            mask_points = np.where(img > 0.5)
        xs = mask_points[1]
        ys = mask_points[0]
        mask_points_all.append(np.stack((xs,ys,np.ones_like(xs))).astype(np.float))
        mask_ims.append(cur_mask)
    return mask_points_all,np.array(mask_ims)

def refine_visual_hull(masks, Ps, scale, center):
    num_cam=masks.shape[0]
    GRID_SIZE=100
    MINIMAL_VIEWS=45 # Fitted for DTU, might need to change for different data.
    im_height=masks.shape[1]
    im_width = masks.shape[2]
    xx, yy, zz = np.meshgrid(np.linspace(-scale, scale, GRID_SIZE), np.linspace(-scale, scale, GRID_SIZE),
                             np.linspace(-scale, scale, GRID_SIZE))
    points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()))
    points = points + center[:, np.newaxis]
    appears = np.zeros((GRID_SIZE*GRID_SIZE*GRID_SIZE, 1))
    for i in range(num_cam):
        proji = Ps[i] @ np.concatenate((points, np.ones((1, GRID_SIZE*GRID_SIZE*GRID_SIZE))), axis=0)
        depths = proji[2]
        proj_pixels = np.round(proji[:2] / depths).astype(np.long)
        relevant_inds = np.logical_and(proj_pixels[0] >= 0, proj_pixels[1] < im_height)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[0] < im_width)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[1] >= 0)
        relevant_inds = np.logical_and(relevant_inds, depths > 0)
        relevant_inds = np.where(relevant_inds)[0]

        cur_mask = masks[i] > 0.5
        relmask = cur_mask[proj_pixels[1, relevant_inds], proj_pixels[0, relevant_inds]]
        relevant_inds = relevant_inds[relmask]
        appears[relevant_inds] = appears[relevant_inds] + 1

    final_points = points[:, (appears >= MINIMAL_VIEWS).flatten()]
    centroid=final_points.mean(axis=1)
    normalize = final_points - centroid[:, np.newaxis]

    return centroid,np.sqrt((normalize ** 2).sum(axis=0)).mean() * 3,final_points.T

# the normaliztion script needs a set of 2D object masks and camera projection matrices (P_i=K_i[R_i |t_i] where [R_i |t_i] is world to camera transformation)
def get_normalization_function(Ps,mask_points_all,number_of_normalization_points,number_of_cameras,masks_all):
    P_0 = Ps[0]
    Fs = get_fundamental_matrices(P_0, Ps)
    P_0_center = np.linalg.svd(P_0)[-1][-1, :]
    P_0_center = P_0_center / P_0_center[3]

    # Use image 0 as a references
    xs = mask_points_all[0][0, :]
    ys = mask_points_all[0][1, :]

    counter = 0
    all_Xs = []

    # sample a subset of 2D points from camera 0
    indss = np.random.permutation(xs.shape[0])[:number_of_normalization_points]

    for i in indss:
        curx = xs[i]
        cury = ys[i]
        # for each point, check its min/max depth in all other cameras.
        # If there is an intersection of relevant depth keep the point
        observerved_in_all = True
        max_d_all = 1e10
        min_d_all = 1e-10
        for j in range(1, number_of_cameras, 5):
            min_d, max_d = get_min_max_d(curx, cury, Ps[j], mask_points_all[j], P_0, Fs[j], j)

            if abs(min_d) < 0.00001:
                observerved_in_all = False
                break
            max_d_all = np.min(np.array([max_d_all, max_d]))
            min_d_all = np.max(np.array([min_d_all, min_d]))
            if max_d_all < min_d_all + 1e-2:
                observerved_in_all = False
                break
        if observerved_in_all:
            direction = np.linalg.inv(P_0[:3, :3]) @ np.array([curx, cury, 1.0])
            all_Xs.append(P_0_center[:3] + direction * min_d_all)
            all_Xs.append(P_0_center[:3] + direction * max_d_all)
            counter = counter + 1

    print("Number of points:%d" % counter)
    centroid = np.array(all_Xs).mean(axis=0)
    # mean_norm=np.linalg.norm(np.array(allXs)-centroid,axis=1).mean()
    scale = np.array(all_Xs).std()

    # OPTIONAL: refine the visual hull
    centroid,scale,all_Xs = refine_visual_hull(masks_all, Ps, scale, centroid)

    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = centroid[0]
    normalization[1, 3] = centroid[1]
    normalization[2, 3] = centroid[2]

    normalization[0, 0] = scale
    normalization[1, 1] = scale
    normalization[2, 2] = scale
    return normalization,all_Xs


def get_normalization(source_dir, use_linear_init=False):
    print('Preprocessing', source_dir)

    if use_linear_init:
        #Since there is noise in the cameras, some of them will not apear in all the cameras, so we need more points
        number_of_normalization_points=1000
        cameras_filename = "cameras_linear_init"
    else:
        number_of_normalization_points = 100
        cameras_filename = "cameras"

    masks_dir='{0}/mask'.format(source_dir)
    cameras=np.load('{0}/{1}.npz'.format(source_dir, cameras_filename))

    mask_points_all,masks_all=get_all_mask_points(masks_dir)
    number_of_cameras = len(masks_all)
    Ps = get_Ps(cameras, number_of_cameras)

    normalization,all_Xs=get_normalization_function(Ps, mask_points_all, number_of_normalization_points, number_of_cameras,masks_all)

    cameras_new={}

    K = np.eye(4)
    K[0][0] = masks_all.shape[2]/2
    K[0][2] = masks_all.shape[2]/2
    K[1][1] = masks_all.shape[1]/2
    K[1][2] = masks_all.shape[1]/2
    K = np.linalg.inv(K)
    for i in range(number_of_cameras):
        cameras_new['scale_mat_%d'  % i] = normalization
        cameras_new['world_mat_%d'  % i] = np.concatenate((Ps[i],np.array([[0,0,0,1.0]])),axis=0).astype(np.float32)
        cameras_new['camera_mat_%d' % i] = K

    np.savez('{0}/{1}.npz'.format(source_dir, cameras_filename), **cameras_new)
    print('model matrix:')
    
    Mdict = np.array({'model_matrix': normalization})
    npy_name = os.path.join(source_dir,'model_matrix.npy')
    np.save(npy_name, Mdict, allow_pickle=True)

    print(normalization)
    print('--------------------------------------------------------')

    if False: #for debugging
        for i in range(number_of_cameras):
            plt.figure()

            plt.imshow(mpimg.imread('%s/%03d.png' % (masks_path, i)))
            xy = (Ps[i,:2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T)) / (
                        Ps[i,2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T))

            plt.plot(xy[0, :], xy[1, :], '*')
            plt.show()

def preprocess_cameras(source_dir='',dtu=False, use_linear_init=False):
    if dtu:
        source_dir = '../data/DTU'
        scene_dirs = sorted(glob(os.path.join(source_dir, "scan*")))
        for scene_dir in scene_dirs:
            get_normalization(scene_dir,use_linear_init)
    else:
        get_normalization(source_dir, use_linear_init)

    print('Done!')