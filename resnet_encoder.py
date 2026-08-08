# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ProDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from ProDepth.layers import *
from ProDepth.networks.depth_encoder import LiteMono, LiteCVEncoder

from collections import OrderedDict
from ProDepth.layers import ConvBlock, Conv3x3, upsample
from ProDepth import datasets, networks

import math

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        # 修改首层卷积输入通道数
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)# 支持多帧输入
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoderMatching(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, num_layers, pretrained, encoder, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=20.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear',batch_size=12):

        super(ResnetEncoderMatching, self).__init__()

        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True
        self.encoder = encoder
        self.batch_size = 128 # 有改动
        if self.encoder == 'resnet':
            self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        elif self.encoder == 'lite':
            self.num_ch_enc = np.array([64, 64, 128, 224])
        elif self.encoder == 'lvt':
            self.num_ch_enc = np.array([64, 64, 160, 256])

        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None
        if self.encoder == 'resnet':
            resnets = {18: models.resnet18,
                    34: models.resnet34,
                    50: models.resnet50,
                    101: models.resnet101,
                    152: models.resnet152}

            if num_layers not in resnets:
                raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

            encoder = resnets[num_layers](pretrained)
            self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
            self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)
            self.layer2 = encoder.layer2
            self.layer3 = encoder.layer3
            self.layer4 = encoder.layer4

            if num_layers > 34:
                self.num_ch_enc[1:] *= 4

        """Layer to transform a depth image into a point cloud
            """
        # 用于match_features
        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                            height=self.matching_height,
                                            width=self.matching_width)
        # 用于compute_dynamic_flow
        self.backprojector1 = BackprojectDepth1(batch_size=self.batch_size,
                                              height=self.matching_height,
                                              width=self.matching_width)
        """Layer which projects 3D points into a camera with intrinsics K and at position T
            """
        # 用于match_features
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                height=self.matching_height,
                                width=self.matching_width)
        # 用于compute_dynamic_flow
        self.projector1 = Project3D1(batch_size=self.batch_size,
                                   height=self.matching_height,
                                   width=self.matching_width)

        self.compute_depth_bins(min_depth_bin, max_depth_bin)

        self.prematching_conv = nn.Sequential(nn.Conv2d(64, out_channels=16,
                                                        kernel_size=1, stride=1, padding=0),
                                            nn.ReLU(inplace=True)
                                            )

        self.reduce_conv = nn.Sequential(nn.Conv2d(self.num_ch_enc[1] + self.num_depth_bins,
                                                   out_channels=self.num_ch_enc[1],
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)
                                         )


        self.cv_encoder = LiteCVEncoder(model='lite-mono-8m', drop_path_rate=0.4)
        self.load_cv_pretrain()
        self.cv_decoder = networks.depth_decoder1.DepthDecoder(self.cv_encoder.num_ch_enc, [0, 1, 2])

        if self.encoder == 'lite':
            self.lite_encoder = LiteMono(model='lite-mono-8m', drop_path_rate=0.4)
            self.load_pretrain()

    def compute_dynamic_flow(self, lookup_pose, _flow, depth, _K, _invK, current_image, lookup_images):
        flow = _flow[:depth.size(0)]
        world_points_depth = self.backprojector1(depth, _invK)
        _, pix_locs_depth = self.projector1(world_points_depth, _K, lookup_pose[:,0])
        pix_locs_depth = pix_locs_depth.permute(0, 3, 1, 2)
        # --------------- backprojector_batch
        pix_coords = self.backprojector1.pix_coords.view(self.batch_size, 3, \
            self.matching_height, self.matching_width)[:, :2, :, :]
        pix_coords =  pix_coords[:flow.size(0)]
        normal_static_flow = pix_locs_depth - pix_coords # 得到静态流
        dynamic_flow = flow - normal_static_flow # 得到动态流
        # dynamic_flow = flow  # 直接用总光流
        check_flow = torch.norm(dynamic_flow, dim=1, keepdim=True)
        threshold = 3 # 硬阈值
        beta = 1.0 # 软阈值
        segmentation = torch.sigmoid(beta * (check_flow - threshold)) # 动态软掩码，越趋于1越是动态
        # segmentation =  check_flow > threshold # 动态硬掩码，1是动态
        flow_bwd, seg_ref = self.invert_flow(normal_static_flow, segmentation) # 得到反向光流、参考帧上的动态掩码
        static_reference, dynamic_reference = self.warping(lookup_images, current_image, flow_bwd, seg_ref)
        # 得到带补丁的新参考帧，以及该补丁
        return static_reference, dynamic_reference

    def invert_flow(self, fwd_flow, segmentation):
        # Get the backward optical flow.
        B, _, H, W = fwd_flow.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(fwd_flow.device)  # Shape (2, H, W)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        coords = grid + fwd_flow  # Shape (B, 2, H, W)
        bwd_flow = torch.zeros_like(fwd_flow)
        seg_ref = torch.zeros_like(segmentation)
        coords = torch.round(coords).long()
        grid = grid.long()
        coords[:, 0].clamp_(0, W - 1)
        coords[:, 1].clamp_(0, H - 1)
        for b in range(B):
            bwd_flow[b, :, coords[b, 1], coords[b, 0]] = - fwd_flow[b, :, grid[b,1], grid[b,0]]
            seg_ref[b, :, coords[b, 1], coords[b, 0]] = segmentation[b, :, grid[b,1], grid[b,0]]
        return bwd_flow, seg_ref

    def warping(self, lookup_images, current_image, flow_bwd, seg_ref):
        ref_image = F.interpolate(lookup_images[:, 0], scale_factor=1/4, mode='bilinear', align_corners=False)
        cur_image = F.interpolate(current_image, scale_factor=1/4, mode='bilinear', align_corners=False)
        B, _, H, W = flow_bwd.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(flow_bwd.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        new_coords = grid + flow_bwd
        new_coords[:, 0, :, :] = (new_coords[:, 0, :, :] / (W - 1)) * 2 - 1
        new_coords[:, 1, :, :] = (new_coords[:, 1, :, :] / (H - 1)) * 2 - 1
        new_coords = new_coords.permute(0, 2, 3, 1)  # Shape (1, H, W, 2)
        static_reference_dyn = F.grid_sample(cur_image, new_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        # new_static_reference = ref_image*(~seg_ref) + static_reference_dyn*seg_ref # 硬掩码
        new_static_reference = ref_image*(1.0-seg_ref) + static_reference_dyn*seg_ref # 软掩码
        dynamic_reference = static_reference_dyn*seg_ref
        return new_static_reference, dynamic_reference

    def compute_depth_bins(self, min_depth_bin, max_depth_bin):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)计算用于构建成本量的深度分箱。箱将取决于self.depth_binning，深度为线性 （linear） 或反深度 （inverse）"""
        #近距离均匀场景
        if self.depth_binning == 'inverse':
            self.depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                              1 / min_depth_bin,
                                              self.num_depth_bins)[::-1]  # maintain depth order
        #远距离场景
        elif self.depth_binning == 'linear':
            self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)
        #动态范围大的场景
        elif self.depth_binning == 'sid':
            self.depth_bins = np.array(
                [np.exp(np.log(min_depth_bin) + np.log(max_depth_bin / min_depth_bin) * i / (self.num_depth_bins - 1))
                for i in range(self.num_depth_bins)])
        else:
            raise NotImplementedError
        
        self.depth_bins = torch.from_numpy(self.depth_bins).float()

        self.warp_depths = []
        for depth in self.depth_bins:
            depth = torch.ones((1, self.matching_height, self.matching_width)) * depth
            self.warp_depths.append(depth)
        self.warp_depths = torch.stack(self.warp_depths, 0).float()
        if self.is_cuda:
            self.warp_depths = self.warp_depths.cuda()

    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.
        根据 current_feats 和 lookup_feats 之间的 L1 差异计算成本量。

        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).
        我们使用估计的相对姿势、已知的内在函数并使用假设的深度self.warp_depths（深度为线性或反深度为线性）
        将lookup_feats向后扭曲到当前帧中。

        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it
        如果 relative_pose == 0 则表示缺少查找帧（即我们位于序列的开头），因此我们跳过它"""

        batch_cost_volume = []  # store all cost volumes of the batch 存储批次的所有成本量
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence 成本量中“0”的存储位置，以便置信度
        cv_features = []
        
        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width) # 定义代价体的维度结构
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device) # 初始化代价体存储容器
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device) # 初始化有效匹配次数计数器
            
            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1] # 从批量数据中提取当前样本的参考帧特征
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1] # 提取当前样本的参考帧相对位姿

            _K = K[batch_idx:batch_idx + 1] # 提取当前样本的相机内参矩阵
            _invK = invK[batch_idx:batch_idx + 1] # 提取当前样本的逆相机内参矩阵
            world_points = self.backprojector(self.warp_depths, _invK) # 执行反投影计算3D点云

            # loop through ref images adding to the current cost volume 循环遍历添加到当前成本量的ref图像
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx] # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx] # 从批量数据中提取第lookup_idx个参考帧的特征和位姿

                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1]) # 为每个深度假设复制特征
                pix_locs = self.projector(world_points, _K, lookup_pose) # 3D到2D投影计算(D, H, W, 2)
                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                       align_corners=True) # 出形状 (D, C, H, W) 双线性特征采样
                
                # mask values landing outside the image (and near the border)蒙版值位于图像外部（且靠近边界）
                # we want to ignore edge pixels of the lookup images and the current image
                # 我们想忽略查找图像和当前图像的边缘像素
                # because of zero padding in ResNet 因为ResNet中的填充为零
                # Masking of ref image borderref 图像边框的遮罩
                # 参考帧掩码（防止边缘采样）
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                    self.matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

                edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                edge_mask = edge_mask.float()

                # masking of current image 当前帧掩码（排除边缘2像素）
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask

                diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(
                    1) * edge_mask # 特征差异计算

                # integrate into cost volume
                cost_volume = cost_volume + diffs # 累加差异值
                counts = counts + (diffs > 0).float() # 记录有效匹配次数
                
            # average over lookup images
            cost_volume = cost_volume / (counts + 1e-7) # 对多视图匹配结果进行平均  归一化代价体
            cv_feature = cost_volume.clone().detach() # 代价体特征提取 保存未经缺失处理的原始代价体 用于后续可视化或不需要梯度回传的模块

            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            # 如果像素位置缺少一些值（即一些深度落在外面），则设置为 Max of existing values
            missing_val_mask = (cost_volume == 0).float() # 通过cost_volume == 0检测未被任何参考视图覆盖的位置转换为浮点型掩码（1表示缺失，0表示有效）
            if self.set_missing_to_max: # 缺失值填充
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                    cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            # 结果保存
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)
            
            cv_features.append(cv_feature)
            
        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)
        
        cv_features = torch.stack(cv_features,0)

        return batch_cost_volume, cost_volume_masks, cv_features

    def feature_extraction(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet在图像上运行特征提取 - ResNet 的前 2 个块"""

        image = (image - 0.45) / 0.225  # imagenet normalisation
        feats_0 = self.layer0(image)
        feats_1 = self.layer1(feats_0)

        if return_all_feats:
            return [feats_0, feats_1]
        else:
            return feats_1

    def indices_to_disparity(self, indices):
        """Convert cost volume indices to 1/depth for visualisation将成本体积指数转换为 1/depth 以进行可视化"""

        batch, height, width = indices.shape
        depth = self.depth_bins[indices.reshape(-1).cpu()]
        disp = 1 / depth.reshape((batch, height, width))
        return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed
        根据深度箱的观察次数返回 'confidence' 掩码"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask

    def forward(self, current_image, lookup_images, poses, flow, depth, K, invK,
                min_depth_bin=None, max_depth_bin=None, mono_disp=None, var = None,using_flow=False, epoch=100
                ):
        #+++++++++++++++++++++++++++++++++++++++++++++++++#
        if self.encoder == 'resnet':
            # feature extraction
            self.features = self.feature_extraction(current_image, return_all_feats=True)
            current_feats = self.features[-1]

            """^^^^^^^^Static Flow Extraction^^^^^^^^"""
            #if using_flow and epoch > 1:
            if using_flow:
                # this one is usually set as half of freeze epoch
                with torch.no_grad():
                    alpha = 0.3
                    static_reference = self.compute_dynamic_flow(poses, flow, depth, K, invK, current_image,
                                                                 lookup_images)
                    static_reference = F.interpolate(static_reference, scale_factor=4, mode='bilinear',
                                                     align_corners=False)
                    if len(lookup_images.shape) > 4:
                        static_reference = static_reference[:, None]
                    lookup_images = alpha * static_reference + (1 - alpha) * lookup_images

            """^^^^^^^^Static Flow Extraction^^^^^^^^"""

            # feature extraction on lookup images - disable gradients to save memory
            with torch.no_grad():
                if self.adaptive_bins:
                    self.compute_depth_bins(min_depth_bin, max_depth_bin)

                batch_size, num_frames, chns, height, width = lookup_images.shape
                lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)
                lookup_feats = self.feature_extraction(lookup_images,
                                                    return_all_feats=False)
                _, chns, height, width = lookup_feats.shape
                lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)

                # warp features to find cost volume
                cost_volume, missing_mask, cv_feature = \
                    self.match_features(current_feats, lookup_feats, poses, K, invK)
                confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                            (1 - missing_mask.detach()))
            #+++++++++++++++++++++++++++++++++++++++++++++++++#
            # CV Deocder
            cv_tmp, cv_x, mono_features = self.cv_encoder.forward_features(current_image)
            self.cv_features = self.cv_encoder(cv_tmp, cv_x, mono_features ,cv_feature) 
            self.cv_features = self.cv_decoder(self.cv_features)[('disp',0)].squeeze(1)

        #+++++++++++++++++++++++++++++++++++++++++++++++++#
        elif self.encoder == 'lite':
            tmp, x, self.features = self.lite_encoder.forward_features(current_image)
            # tmp:[第1层下采样，第1层特征] x:[1/2，1/4，1/8，1/16] features:[第0层特征，第1层特征]

            lite_current_feats = self.features[-1]  # [B, C, H, W]

            """^^^^^^^^Static Flow Extraction^^^^^^^^"""
            if using_flow :
                # this one is usually set as half of freeze epoch
                with torch.no_grad():
                    alpha = 0.05
                    # static_reference:带补丁的新参考帧，dynamic_reference:该补丁
                    static_reference, dynamic_reference = self.compute_dynamic_flow(poses, flow, depth, K, invK, current_image,
                                                                 lookup_images)
                    static_reference = F.interpolate(static_reference, scale_factor=4, mode='bilinear',
                                                     align_corners=False)
                    dynamic_reference = F.interpolate(dynamic_reference, scale_factor=4, mode='bilinear',
                                                     align_corners=False)
                    if len(lookup_images.shape) > 4:
                        dynamic_reference = dynamic_reference[:, None]
                    lookup_images = alpha * dynamic_reference + (1 - alpha) * lookup_images
                    # lookup_images:0.95的背景，0.95动态+0.05修正

                    # 保存到外部变量，准备 return
                    out_static_ref = static_reference.clone()
                    out_dynamic_ref = dynamic_reference.clone()
            """^^^^^^^^Static Flow Extraction^^^^^^^^"""

            # feature extraction on lookup images - disable gradients to save memory
            with torch.no_grad():
                if self.adaptive_bins:
                    self.compute_depth_bins(min_depth_bin, max_depth_bin)

                batch_size, num_frames, chns, height, width = lookup_images.shape
                lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)
                _, _, feats = self.lite_encoder.forward_features(lookup_images) # feats:[第0层特征，第1层特征] [B, C, H, W]
                lite_lookup_feats = feats[-1] # 第1层特征
                _, chns, height, width = lite_lookup_feats.shape
                lite_lookup_feats = lite_lookup_feats.reshape(batch_size, num_frames, chns, height, width)
                
                # warp features to find cost volume
                # cost_volume:归一化填充缺失处后的代价体 cv_feature:未归一化填充的原代价体
                cost_volume, missing_mask, cv_feature = \
                    self.match_features(lite_current_feats, lite_lookup_feats, poses, K, invK) # missing_mask为代价为0的地方，是无效匹配
                confidence_mask = self.compute_confidence_mask(cost_volume.detach() * # 计算有效匹配的地方且深度值不为0的地方
                                                            (1 - missing_mask.detach())) # 因此confidence_mask要求代价>0，深度值>0，即可信
            #+++++++++++++++++++++++++++++++++++++++++++++++++#
            # CV Decoder
            cv_tmp = [t.clone().detach() for t in tmp]
            cv_x = [t.clone().detach() for t in x]
            mono_features = [t.clone().detach() for t in self.features]
            # cv_tmp:[第1层下采样，第1层特征] cv_x:[1/2，1/4，1/8，1/16] mono_features:[第0层特征，第1层特征]

            self.cv_features = self.cv_encoder(cv_tmp, cv_x, mono_features ,cv_feature) 
            self.cv_features = self.cv_decoder(self.cv_features)[('disp',0)].squeeze(1) # 即Dcv
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++#
        # CV Refine

        cost_volume_distribution = cost_volume.clone().detach()
        cv_max,_ = torch.max(cost_volume_distribution,1)
        cv_min,_ = torch.min(cost_volume_distribution,1)
        cost_volume_distribution = (cost_volume_distribution - cv_min.unsqueeze(1)) / (cv_max.unsqueeze(1) - cv_min.unsqueeze(1) + 1e-7)
        # 将cost_volume归一化到0~1，便于与mono的高斯分布在同一数量级上进行加权融合

        mono_output = F.interpolate(mono_disp.detach(), [mono_disp.shape[-2] // 4, mono_disp.shape[-1] // 4], mode="bilinear")
        _mono_disp, _mono_depth = disp_to_depth(mono_output,0.1, 100)
        d_i = self.depth_bins.view(1, -1, 1, 1).repeat(_mono_depth.shape[0],1,_mono_depth.shape[-2],_mono_depth.shape[-1]).cuda()
        sigma = F.interpolate(var.detach(), [mono_disp.shape[-2] // 4, mono_disp.shape[-1] // 4], mode="bilinear")
        # # ==================== 【Fixed Variance 消融实验修改】 ====================
        # # 1. 照常获取网络预测的真实 sigma
        # sigma_real = F.interpolate(var.detach(), [mono_disp.shape[-2] // 4, mono_disp.shape[-1] // 4], mode="bilinear")
        # # 2. 计算空间维度 (H, W) 的平均值，保持 batch 和 channel 维度
        # # 这会让 sigma_mean 的形状变成 [B, 1, 1, 1]
        # sigma_mean = sigma_real.mean(dim=(2, 3), keepdim=True)
        # # 3. 将这个平均值扩张回原来的形状 [B, 1, H, W]
        # # 现在的 sigma 整张图所有像素的值都是一模一样的平均值了，数量级绝对正确
        # sigma = sigma_mean.expand_as(sigma_real)
        # # =========================================================================
        gaussian_mono_distribution = (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-((d_i - _mono_depth) ** 2) / (2 * sigma ** 2))
        gmd_max,_ = torch.max(gaussian_mono_distribution,1)
        gmd_min,_ = torch.min(gaussian_mono_distribution,1)
        gaussian_mono_distribution = (gaussian_mono_distribution - gmd_min.unsqueeze(1)) / (gmd_max.unsqueeze(1) - gmd_min.unsqueeze(1) + 1e-7)
        # mono的高斯分布

        cv_output = F.interpolate(self.cv_features.unsqueeze(1).detach(), [mono_disp.shape[-2] // 4, mono_disp.shape[-1] // 4], mode="bilinear")
        matching_disp, matching_depth = disp_to_depth(cv_output.detach(),0.1, 100)
        # 从Dcv中提取真实视差和深度，用于深度一致性掩码的计算

        depth_mask = torch.exp(-torch.abs(matching_depth - _mono_depth)*(2/3))
        disp_mask = torch.exp(-torch.abs(matching_disp - _mono_disp)*(2/3))
        weighted_mask = disp_mask * depth_mask  # 0: Moving, 1: Static 。不确定性概率的补数，即为1-U
        binary_mask = (weighted_mask > 0.4).float() # 源码是0.4

        # # ========================================
        # # 构建“自引导高斯分布 (Self-Guided Gaussian)”
        # # 核心改变：将高斯公式里的均值 _mono_depth 替换为 matching_depth
        # d_i = self.depth_bins.view(1, -1, 1, 1).repeat(matching_depth.shape[0], 1, matching_depth.shape[-2],
        #                                                matching_depth.shape[-1]).cuda()
        # sigma = F.interpolate(var.detach(), [mono_disp.shape[-2] // 4, mono_disp.shape[-1] // 4],
        #                       mode="bilinear")  # 方差仍然用网络预测的，保证控制变量
        # # 注意这里用的是 matching_depth
        # gaussian_self_distribution = (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(
        #     -((d_i - matching_depth) ** 2) / (2 * sigma ** 2))
        # gsd_max, _ = torch.max(gaussian_self_distribution, 1)
        # gsd_min, _ = torch.min(gaussian_self_distribution, 1)
        # # 变量名改为了 gaussian_self_distribution
        # gaussian_self_distribution = (gaussian_self_distribution - gsd_min.unsqueeze(1)) / (
        #             gsd_max.unsqueeze(1) - gsd_min.unsqueeze(1) + 1e-7)
        # # ========================================

        # # ==================== 【Fixed Weight 消融实验修改】 ====================
        # # 强制将融合指数替换为固定常数 0.5
        # # 意思是：不管这个区域是极度混乱还是轻微混乱，我统统给单目和多帧各 50% 的权重
        # fixed_weight = 0.5
        # fusion_weight = torch.ones_like(weighted_mask) * fixed_weight
        # # ==================================================================

        softmax_cv_distribution = torch.softmax(-cost_volume_distribution,1)
        sfm_cvd_max,_ = torch.max(softmax_cv_distribution,1)
        sfm_cvd_min,_ = torch.min(softmax_cv_distribution,1)
        softmax_cv_distribution = (softmax_cv_distribution - sfm_cvd_min.unsqueeze(1)) / (sfm_cvd_max.unsqueeze(1) - sfm_cvd_min.unsqueeze(1) + 1e-7)
        # 代价体的概率分布

        # refined_cost_volume_distribution = (softmax_cv_distribution**(weighted_mask) * gaussian_mono_distribution**(1-weighted_mask))
        refined_cost_volume_distribution = (softmax_cv_distribution*weighted_mask) + (gaussian_mono_distribution*(1-weighted_mask))
        # refined_cost_volume_distribution = (softmax_cv_distribution**(fusion_weight) * gaussian_mono_distribution**(1-fusion_weight))
        # refined_cost_volume_distribution = (softmax_cv_distribution**(weighted_mask) * gaussian_self_distribution**(1-weighted_mask))
        refined_cost_volume_distribution = binary_mask * softmax_cv_distribution + (1-binary_mask) * refined_cost_volume_distribution
        # 融合后的概率分布，加了个binary_mask，使得动态概率＞0.6时才修正代价体的概率，binary_mask是门控，weighted_mask是融合权重

        refined_max, _ = torch.max(refined_cost_volume_distribution,1)
        refined_min, _ = torch.min(refined_cost_volume_distribution,1)
        refined_cost_volumes = ((refined_max.unsqueeze(1) - refined_cost_volume_distribution) / (refined_max.unsqueeze(1) - refined_min.unsqueeze(1) + 1e-7)) * (cv_max.unsqueeze(1)-cv_min.unsqueeze(1)) + cv_min.unsqueeze(1)
        # 修正后的代价体

        maxs, argmax = torch.max(refined_cost_volumes, 1)
        refined_lowest_cost = self.indices_to_disparity(argmax)

        maxs, argmax = torch.max(gaussian_mono_distribution, 1)
        # maxs, argmax = torch.max(gaussian_self_distribution, 1)
        gaussian_cost = self.indices_to_disparity(argmax)
        # 这俩是可视化调试用的
        #+++++++++++++++++++++++++++++++++++++++++++++++++#

        # for visualisation - ignore 0s in cost volume for minimum
        viz_cost_vol = cost_volume.clone().detach()
        viz_cost_vol[viz_cost_vol == 0] = 100
        mins, argmin = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argmin)

        # mask the cost volume based on the confidence
        cost_volume *= confidence_mask.unsqueeze(1)

        post_matching_feats = self.reduce_conv(torch.cat([self.features[-1], refined_cost_volumes], 1))

        if self.encoder == 'resnet':
            self.features.append(self.layer2(post_matching_feats))
            self.features.append(self.layer3(self.features[-1]))
            self.features.append(self.layer4(self.features[-1]))

        elif self.encoder == "lite":
            tmp[-1] = post_matching_feats
            self.features = self.lite_encoder.forward_features2(tmp, x, self.features)
            
        return self.features, lowest_cost, confidence_mask, self.cv_features, binary_mask, refined_lowest_cost, \
            gaussian_cost, lookup_images, out_static_ref, out_dynamic_ref
        # self.features:修正后的代价体与目标帧经过lite编码器得到的特征
        # self.cv_features:通过代价体得到的辅助深度图Dcv
        # confidence_mask:代价>0，深度值>0的地方，用于可视化
        # binary_mask:网络认为的静态区域
        # lowest_cost、refined_lowest_cost、gaussian_cost:用于可视化调试的最小代价像素点

    def cuda(self):
        super().cuda()
        self.backprojector.cuda()
        self.projector.cuda()
        self.backprojector1.cuda()
        self.projector1.cuda()
        self.is_cuda = True
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cuda()

    def cpu(self):
        super().cpu()
        self.backprojector.cpu()
        self.projector.cpu()
        self.backprojector1.cpu()
        self.projector1.cpu()
        self.is_cuda = False
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cpu()

    def to(self, device):
        if str(device) == 'cpu':
            self.cpu()
        elif str(device) == 'cuda':
            self.cuda()
        else:
            raise NotImplementedError
        
    def load_pretrain(self):
        path = os.path.expanduser("/mnt/harddisk3/Zhangaoqi/codespace/Prodepth/Pretrained/lite-mono-8m-pretrain.pth")
        model_dict = self.lite_encoder.state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.lite_encoder.load_state_dict(model_dict)
        print('MULTI ENCODER loaded.')

    def load_cv_pretrain(self):
        path = os.path.expanduser("/mnt/harddisk3/Zhangaoqi/codespace/Prodepth/Pretrained/lite-mono-8m-pretrain.pth")
        model_dict = self.cv_encoder.state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.cv_encoder.load_state_dict(model_dict)
        print('CV ENCODER loaded.')


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1, **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
