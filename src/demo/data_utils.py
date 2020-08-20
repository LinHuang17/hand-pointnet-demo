"""
utils for data (preprocess), refer to:
"Hand pointnet: 3d hand pose estimation using point sets." Ge, Liuhao, et al. CVPR 2018.
"""

#import pcl
import random
import numpy as np
from numba import jit
from sklearn.decomposition import PCA

pca = PCA()
SAMPLE_NUM = 1024

# @profile
# @jit
def pca_obb(hand_points):
    """
    mainly follow data preprocessing from hand-pointnet
    https://sites.google.com/site/geliuhaontu/home
    """

    pca.fit(hand_points)
    coeff = pca.components_.T
    if coeff[1, 0] < 0:
        coeff[:, 0] = -coeff[:, 0]
    if coeff[2, 2] < 0:
        coeff[:, 2] = -coeff[:, 2]
    coeff[:, 1] = np.cross(coeff[:, 2], coeff[:, 0])
    
    return np.matmul(hand_points, coeff), coeff

# @profile
def normalize_w_sample(hand_points_rotate, size):
    """
    mainly follow data preprocessing from hand-pointnet
    https://sites.google.com/site/geliuhaontu/home
    """

    rand_idx = random.sample(range(size), SAMPLE_NUM)
    hand_points_rotate_sampled = hand_points_rotate[rand_idx, :]

    x_min_max = [min(hand_points_rotate[:, 0]), max(hand_points_rotate[:, 0])]
    y_min_max = [min(hand_points_rotate[:, 1]), max(hand_points_rotate[:, 1])]
    z_min_max = [min(hand_points_rotate[:, 2]), max(hand_points_rotate[:, 2])]

    scale = 1.2
    bb3d_x_len = scale * (x_min_max[1] - x_min_max[0])
    bb3d_y_len = scale * (y_min_max[1] - y_min_max[0])
    bb3d_z_len = scale * (z_min_max[1] - z_min_max[0])
    
    # scale normalization
    max_bb3d_len = bb3d_x_len
    hand_points_normalized_sampled = hand_points_rotate_sampled / max_bb3d_len

    # offset
    offset = np.mean(hand_points_normalized_sampled, axis=0)
    
    return (hand_points_normalized_sampled - np.tile(offset, (SAMPLE_NUM, 1))), rand_idx, max_bb3d_len, offset

# @profile
def get_normals_fast(hand_points):
    cloud = pcl.PointCloud(hand_points.astype(np.float32))
    normals = cloud.make_NormalEstimation()
    normals.set_KSearch(30)
    normals = normals.compute()
    return normals.to_array()[:, :3]

# @profile
def get_normals_fast1(rand_idx, hand_points, coeff):
    # idx = random.sample(range(hand_points.shape[0]), int(hand_points.shape[0]/3))
    cloud = pcl.PointCloud(hand_points.astype(np.float32))
    normals = cloud.make_NormalEstimation()
    normals.set_KSearch(30)
    normals = normals.compute()
    normals = normals.to_array()[rand_idx, :3]
    return np.matmul(normals, coeff)

@jit
def px2cam_w_sample(intr, depth_img):

    bb_height = intr.height
    bb_width = intr.width
    center_x = intr.ppx
    center_y = intr.ppy
    focal_x = intr.fx
    focal_y = intr.fy
    num_px = bb_height * bb_width

    pt_3d = np.zeros((num_px, 3))
    for idx1 in range(bb_height):
        for idx2 in range(bb_width):
            idx = idx2 * bb_height + idx1
            z_value = depth_img[idx1, idx2]
            pt_3d[idx, 0] = -(center_x - idx2) * z_value / focal_x
            pt_3d[idx, 1] = (center_y - idx1) * z_value / focal_y
            pt_3d[idx, 2] = z_value

    sample_pt_3d = pt_3d[np.where((pt_3d[:, 0] != 0) | (pt_3d[:, 1] != 0) | (pt_3d[:, 2] != 0))]
    # random sampling
    rand_idx = random.sample(range(sample_pt_3d.shape[0]), 8000)
    sample_pt_3d = sample_pt_3d[rand_idx, :]
    return sample_pt_3d

@jit
# @profile
def px2cam(intr, depth_img):

    bb_height = intr.height
    bb_width = intr.width
    center_x = intr.ppx
    center_y = intr.ppy
    focal_x = intr.fx
    focal_y = intr.fy
    num_px = bb_height * bb_width

    pt_3d = np.zeros((num_px, 3))
    for idx1 in range(bb_height):
        for idx2 in range(bb_width):
            idx = idx2 * bb_height + idx1
            z_value = depth_img[idx1, idx2]
            pt_3d[idx, 0] = -(center_x - idx2) * z_value / focal_x
            pt_3d[idx, 1] = (center_y - idx1) * z_value / focal_y
            pt_3d[idx, 2] = z_value

    all_pt_3d = pt_3d[np.where((pt_3d[:, 0] != 0) | (pt_3d[:, 1] != 0) | (pt_3d[:, 2] != 0))]
    return all_pt_3d

@jit
def cam2px(intr, pt_3d):
    
    center_x = intr.ppx
    center_y = intr.ppy
    focal_x = intr.fx
    pt_2d = np.zeros((21, 2))

    for idx in range(21):
        pt_2d[idx, 0] = ((pt_3d[idx, 0] * focal_x) / pt_3d[idx, 2]) + center_x
        pt_2d[idx, 1] = center_y - ((pt_3d[idx, 1] * focal_x) / pt_3d[idx, 2])

    return pt_2d.astype(np.int32)