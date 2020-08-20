"""
Simple real-time demo for
"Hand pointnet: 3d hand pose estimation using point sets." Ge, Liuhao, et al. CVPR 2018.
based on
"Pointnet++: Deep hierarchical feature learning on point sets in a metric space." Qi, Charles, et al. NIPS 2017.
https://github.com/charlesq34/pointnet2.git
Input depth maps are captured using
https://github.com/IntelRealSense/librealsense.git
"""

import os
import sys
import time
import argparse
import numpy as np

import tensorflow as tf
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import data_utils
from model import PointNet_Plus
from net_utils1 import group_points
# from net_utils2 import group_points

from numba import jit
import pyrealsense2 as rs 

sys.path.append("/home/demo/src/")
sys.path.append('/home/demo/pointnet2/tf_ops/sampling')
sys.path.append('/home/demo/pointnet2/tf_ops/grouping')
import tf_grouping
import tf_sampling
import cv2
import random

parser = argparse.ArgumentParser()

parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=3, help='number of input point features')
parser.add_argument('--SAMPLE_NUM', type=int, default=1024, help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default=21, help='number of joints')
parser.add_argument('--JOINT_DIM', type=int, default=3, help='dim. of each joint')
parser.add_argument('--PCA_SZ', type=int, default=42, help='number of PCA components')

parser.add_argument('--knn_K', type=int, default=64, help='K for knn search')
parser.add_argument('--sample_num_level1', type=int, default=512, help='number of first layer groups')
parser.add_argument('--sample_num_level2', type=int, default=128, help='number of second layer groups')
parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')

args = parser.parse_args()

JOINT_NUM = args.JOINT_NUM
SAMPLE_NUM = args.SAMPLE_NUM
sample_num_level1 = args.sample_num_level1
sample_num_level2 = args.sample_num_level2

# GPU environment
torch.cuda.set_device(0)
usecuda = torch.cuda.is_available()
device = torch.device("cuda:0" if usecuda else "cpu")

# model
netR = PointNet_Plus(args)
netR.load_state_dict(torch.load('../../model/netR_16.pth'))
netR.cuda()
netR.eval()

try:
    print("Starting Depth Camera")

    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)

    # Start streaming
    pipeline_profile = pipeline.start(config)
    profile = pipeline_profile.get_stream(rs.stream.depth)
    intrinsics = profile.as_video_stream_profile().get_intrinsics()

    tf.InteractiveSession()
    while True:
        try:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            # raw depth maps
            depth_raw = np.array(depth_frame.get_data())
            # convert depth maps to color maps for display
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)

            # convert depth map into point cloud
            all_points = data_utils.px2cam_w_sample(intrinsics, depth_raw)
            # hand segmentation from the input 3D scene
            # depth-based segmentation, assume hand is the cloest object to the camera
            hand_points = all_points[all_points[:, 2] < 800]

            # OBB-based hand points preprocessing 
            hand_points_rotate, coeff = data_utils.pca_obb(hand_points)
            hand_points_normalized_sampled, rand_idx, volume_length, offset = data_utils.normalize_w_sample(
                                                            hand_points_rotate, hand_points.shape[0])

            # FPS: farthest point sampling
            # 1st leveln
            hand_point = tf.convert_to_tensor(
                hand_points_normalized_sampled[np.newaxis, :], dtype=tf.float32)
            sampled_idx_l1 = tf_sampling.farthest_point_sample(
                sample_num_level1, hand_point).eval().squeeze()
            other_idx = np.setdiff1d(np.arange(SAMPLE_NUM), sampled_idx_l1)
            new_idx = np.concatenate((sampled_idx_l1, other_idx))
            hand_points_normalized_sampled = hand_points_normalized_sampled[new_idx,:]
            
            # 2nd level
            hand_point = tf.convert_to_tensor(
                hand_points_normalized_sampled[:sample_num_level1, :3][np.newaxis, :], dtype=tf.float32)
            sampled_idx_l2 = tf_sampling.farthest_point_sample(
                sample_num_level2, hand_point).eval().squeeze()
            other_idx = np.setdiff1d(np.arange(sample_num_level1), sampled_idx_l2)
            new_idx = np.concatenate((sampled_idx_l2, other_idx))
            hand_points_normalized_sampled[:sample_num_level1, :] = hand_points_normalized_sampled[new_idx, :]
            points = torch.from_numpy(np.expand_dims(
                hand_points_normalized_sampled.astype(np.float32), axis=0)).cuda()

            # forward pass
            inputs_level1, inputs_level1_center = group_points(points, args)
            with torch.no_grad():
                inputs_level1, inputs_level1_center = Variable(inputs_level1), Variable(inputs_level1_center)
                out = netR(inputs_level1, inputs_level1_center)
            est_joint_obb = out.data
            est_joint_obb = est_joint_obb.view(JOINT_NUM, JOINT_DIM).cpu().numpy()

            est_joint_obb_offset = est_joint_obb + np.tile(offset, (JOINT_NUM, 1))
            est_joint_obb_offset_rescale = est_joint_obb_offset * volume_length
            est_joint_cam = np.matmul(est_joint_obb_offset_rescale, coeff.T)
            est_joint_px = data_utils.cam2px(intrinsics, est_joint_cam)

            for idx in range(JOINT_NUM):
                depth_colormap = cv2.circle(depth_colormap, (est_joint_px[idx,0], est_joint_px[idx,1]), 5, (0,0,255), -1)
            cv2.namedWindow('Pose Estimation', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Pose Estimation', depth_colormap)
            cv2.waitKey(1)

        except (ValueError,IndexError):
            print("Cannot find hand")
finally:
    pipeline.stop()
