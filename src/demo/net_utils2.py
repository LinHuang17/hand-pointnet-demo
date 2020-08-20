"""
utils for grouping operations, refer to:
"Hand pointnet: 3d hand pose estimation using point sets." Ge, Liuhao, et al. CVPR 2018.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from numba import jit 

@jit
def funn(idx, inputs1_idx, invalid_map):
    for r in range(idx):
        for i in range(64):
            if invalid_map[:,r,:][0,i] == 1:
                inputs1_idx[:,r,:][0,i] = r 
    return inputs1_idx

# # @profile
def group_points(points, opt):
    # group points using knn and ball query
    # points: B * 1024 * 6
    # len(points)
    cur_train_size = len(points)
    inputs1_diff = points[:, :, 0:3].transpose(1, 2).unsqueeze(1).expand(cur_train_size, opt.sample_num_level1, 3, opt.SAMPLE_NUM) \
        - points[:, 0:opt.sample_num_level1, 0:3].unsqueeze(-1).expand(cur_train_size, opt.sample_num_level1, 3, opt.SAMPLE_NUM)  # B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024
    # dists: B * 512 * 64; inputs1_idx: B * 512 * 64
    dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False, sorted=False)

    # ball query
    # anything less than ball_radius is invalid in the 1-0 invalid matrix
    invalid_map = dists.gt(opt.ball_radius)  # B * 512 * 64
    # print(inputs1_idx.shape, invalid_map.shape)
    a, b = inputs1_idx, invalid_map
    inputs1_idx = torch.from_numpy(funn(512,inputs1_idx.cpu().numpy(), invalid_map.cpu().numpy())).cuda()

    idx_group_l1_long = inputs1_idx.view(cur_train_size, opt.sample_num_level1 * opt.knn_K, 1).expand(cur_train_size, opt.sample_num_level1 * opt.knn_K, opt.INPUT_FEATURE_NUM)
    inputs_level1 = points.gather(1, idx_group_l1_long).view(cur_train_size, opt.sample_num_level1, opt.knn_K, opt.INPUT_FEATURE_NUM)  # B*512*64*6

    inputs_level1_center = points[:, 0:opt.sample_num_level1, 0:3].unsqueeze(2)       # B*512*1*3
    inputs_level1[:, :, :, 0:3] = inputs_level1[:, :, :, 0:3] - \
        inputs_level1_center.expand(cur_train_size, opt.sample_num_level1, opt.knn_K, 3)
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1, 4).squeeze(4)  # B*6*512*64
    inputs_level1_center = inputs_level1_center.contiguous().view(-1, 1, opt.sample_num_level1, 3).transpose(1, 3)  # B*3*512*1
    return inputs_level1, inputs_level1_center
    # inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1

# @jit
# @profile
def group_points_2(points, sample_num_level1, sample_num_level2, knn_K, ball_radius):
    # group points using knn and ball query
    # points: B*(3+128)*512
    cur_train_size = points.size(0)
    inputs1_diff = points[:, 0:3, :].unsqueeze(1).expand(cur_train_size, sample_num_level2, 3, sample_num_level1) \
        - points[:, 0:3, 0:sample_num_level2].transpose(1, 2).unsqueeze(-1).expand(cur_train_size, sample_num_level2, 3, sample_num_level1)  # B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512
    # dists: B * 128 * 64; inputs1_idx: B * 128 * 64
    dists, inputs1_idx = torch.topk(
        inputs1_diff, knn_K, 2, largest=False, sorted=False)

    # ball query
    # B * 128 * 64, invalid_map.float().sum()
    invalid_map = dists.gt(ball_radius)
    # pdb.set_trace()
    inputs1_idx = torch.from_numpy(funn(128,inputs1_idx.cpu().numpy(), invalid_map.cpu().numpy())).cuda()

    idx_group_l1_long = inputs1_idx.view(cur_train_size, 1, sample_num_level2 * knn_K).expand(
        cur_train_size, points.size(1), sample_num_level2 * knn_K)
    inputs_level2 = points.gather(2, idx_group_l1_long).view(
        cur_train_size, points.size(1), sample_num_level2, knn_K)  # B*131*128*64

    inputs_level2_center = points[:, 0:3, 0:sample_num_level2].unsqueeze(
        3)       # B*3*128*1
    inputs_level2[:, 0:3, :, :] = inputs_level2[:, 0:3, :, :] - \
        inputs_level2_center.expand(
            cur_train_size, 3, sample_num_level2, knn_K)  # B*3*128*64
    return inputs_level2, inputs_level2_center
    # inputs_level2: B*131*sample_num_level2*knn_K, inputs_level2_center: B*3*sample_num_level2*1
    