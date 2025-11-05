import torch
import torch.nn as nn
from .utils.spconv_utils import replace_feature, spconv
from .utils import common_utils
from .spconv_backbone import post_act_block
from functools import partial
from models.modules.sparse_instancenorm import SparseInstanceNorm

RELU_FN = partial(nn.LeakyReLU, 0.01, inplace=True)


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = RELU_FN()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out).features)
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out).features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self,
                 input_channels, grid_size, voxel_size,
                 point_cloud_range, width: float = 2,
                 skip_last_invconv=False,
                 **kwargs):
        super().__init__()
        
        self.sparse_shape = grid_size[::-1]# + [1, 0, 0]
        self.voxel_size = voxel_size
        self.skip_last_invconv = skip_last_invconv
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(SparseInstanceNorm, eps=1e-3, affine=True)
        self.norm_fn = norm_fn
        w = width
        block = partial(post_act_block)

        self.conv_input = spconv.SparseSequential(
            block(input_channels, int(16 * w), 3, norm_fn=norm_fn, indice_key='subm1'),
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(int(16 * w), int(16 * w), norm_fn=norm_fn, indice_key='subm1'),
            SparseBasicBlock(int(16 * w), int(16 * w), norm_fn=norm_fn, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(int(16 * w), int(32 * w), 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(int(32 * w), int(32 * w), 3, norm_fn=norm_fn, indice_key='subm2'),
            SparseBasicBlock(int(32 * w), int(32 * w), 3, norm_fn=norm_fn, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(int(32 * w), int(64 * w), 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(int(64 * w), int(64 * w), 3, norm_fn=norm_fn, indice_key='subm3'),
            SparseBasicBlock(int(64 * w), int(64 * w), 3, norm_fn=norm_fn, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(int(64 * w), int(64 * w), 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(int(64 * w), int(64 * w), 3, norm_fn=norm_fn, indice_key='subm4'),
            SparseBasicBlock(int(64 * w), int(64 * w), 3, norm_fn=norm_fn, indice_key='subm4'),
        )

        self.conv5 = spconv.SparseSequential(
            block(int(64 * w), int(128 * w), 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(int(128 * w), int(128 * w), 3, norm_fn=norm_fn, indice_key='subm5'),
            SparseBasicBlock(int(128 * w), int(128 * w), 3, norm_fn=norm_fn, indice_key='subm5'),
        )

        # decoder
        self.conv_up_t5 = SparseBasicBlock(int(128 * w), int(128 * w), indice_key='subm5', norm_fn=norm_fn)
        self.conv_up_m5 = block(int(256 * w), int(128 * w), 3, norm_fn=norm_fn, padding=1, indice_key='subm5')
        self.inv_conv5 = block(int(128 * w), int(64 * w), 3, norm_fn=norm_fn, indice_key='spconv5', conv_type='inverseconv')

        self.conv_up_t4 = SparseBasicBlock(int(64 * w), int(64 * w), indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(int(128 * w), int(64 * w), 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(int(64 * w), int(64 * w), 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        self.conv_up_t3 = SparseBasicBlock(int(64 * w), int(64 * w), indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(int(128 * w), int(64 * w), 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(int(64 * w), int(32 * w), 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        self.conv_up_t2 = SparseBasicBlock(int(32 * w), int(32 * w), indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(int(64 * w), int(32 * w), 3, norm_fn=norm_fn, padding=1, indice_key='subm2')
        self.inv_conv2 = block(int(32 * w), int(16 * w), 3, norm_fn=norm_fn, indice_key='spconv2',
                               conv_type='inverseconv')

        self.conv_up_t1 = SparseBasicBlock(int(16 * w), int(16 * w), indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(int(32 * w), int(16 * w), 3, norm_fn=norm_fn, indice_key='subm1')

        self.final_conv = spconv.SparseSequential(
            SparseBasicBlock(int(16 * w), int(16 * w), 3, norm_fn=norm_fn, indice_key='subm1')
        )
        self.num_point_features = int(16 * w)

        print(f">>>> Param count: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M")


    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, voxel_coords, voxel_features, batch_size, collect_stats=False, save_low_res_feats=False):
        """
        Args:
            batch_dict:
                batch_size: int
                voxel_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, x_idx, y_idx, z_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        if voxel_features.ndim == 1:
            voxel_features = voxel_features.unsqueeze(-1)

        # tranpose voxel_coords to [batch_idx, z_idx, y_idx, x_idx]
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        assert voxel_features.dtype == torch.float32
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        # for segmentation head
        x_up5 = self.UR_block_forward(x_conv5, x_conv5, self.conv_up_t5, self.conv_up_m5, self.inv_conv5)
        x_up4 = self.UR_block_forward(x_conv4, x_up5, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        
        if save_low_res_feats:
            self.low_res_feats = x_up4

        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        
        if self.skip_last_invconv:
            return x_up3

        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.final_conv)

        # if collect_stats:
        #     stats = {
        #         'x_shape': x.features.shape,
        #         'x_conv1_shape': x_conv1.features.shape,
        #         'x_conv2_shape': x_conv2.features.shape,
        #         'x_conv3_shape': x_conv3.features.shape,
        #         'x_conv4_shape': x_conv4.features.shape,
        #         'x_up4_shape': x_up4.features.shape,
        #         'x_up3_shape': x_up3.features.shape,
        #         'x_up2_shape': x_up2.features.shape,
        #         'x_up1_shape': x_up1.features.shape,
        #         'x_conv1_spatial_shape': x_conv1.spatial_shape,
        #         'x_conv2_spatial_shape': x_conv2.spatial_shape,
        #         'x_conv3_spatial_shape': x_conv3.spatial_shape,
        #         'x_conv4_spatial_shape': x_conv4.spatial_shape,
        #         'x_up4_spatial_shape': x_up4.spatial_shape,
        #         'x_up3_spatial_shape': x_up3.spatial_shape,
        #         'x_up2_spatial_shape': x_up2.spatial_shape,
        #         'x_up1_spatial_shape': x_up1.spatial_shape,
        #     }
        #     return stats

        # batch_dict['point_features'] = x_up1.features
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)

        return x_up1