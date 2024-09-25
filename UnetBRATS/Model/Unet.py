import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep
from torch.nn import LayerNorm

from monai.networks.layers import DropPath, Conv

from collections import OrderedDict
from typing import Sequence, Type


class opmoudle(nn.Module):
    def __init__(self):
      super(opmoudle, self).__init__()
      self.a=nn.Parameter(torch.zeros(1))
    def forward(self, or_pre,mm_pre):
        op_pre=or_pre+(1-self.a)*mm_pre
        return op_pre


def init_weights(self, init_type=None):
    for m in self.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up= nn.ConvTranspose3d(in_channels=in_size,
                                          out_channels=in_size,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )
        self.bn1 = nn.BatchNorm3d(num_features=in_size, )
        self.relu = nn.ReLU(inplace=True)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        outputs2=self.bn1(outputs2)
        outputs2 = self.relu(outputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNet(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True,mun_pro=16):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.mun_pro = mun_pro

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.representation = nn.Sequential(
            nn.Conv3d(filters[0], self.mun_pro, 3, padding=1, bias=False),
            nn.BatchNorm3d(self.mun_pro),
            nn.ReLU(),
            nn.Conv3d(self.mun_pro, self.mun_pro, 1)
        )

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        ##
        icl_in_chans = (64, 32)
        icl_in_resolutions = ([24, 24, 24], [48, 48, 48])
        self.dynamic = Dynamicinteraction(
            in_chans=icl_in_chans,
            depths=(2, 2),
            input_resolution=icl_in_resolutions,
            num_classes=n_classes,
            norm_layer=nn.LayerNorm,
            num_pro=self.mun_pro,
        )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs,mm=False):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # print(maxpool1.shape)2, 16, 48, 48, 48

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # print(maxpool2.shape)[2, 32, 24, 24, 24

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # print(maxpool3.shape)torch.Size([2, 64, 12, 12, 12])

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        #print(maxpool4.shape)torch.Size([2, 128, 6, 6, 6])

        center = self.center(maxpool4)
        center = self.dropout1(center)
        #print(center.shape)torch.Size([2, 256, 6, 6, 6])
        up4 = self.up_concat4(conv4, center)
        #print(up4.shape)2, 128, 12, 12, 12
        up3 = self.up_concat3(conv3, up4)#
        #print(up3.shape) 2, 64, 24, 24, 24
        up2 = self.up_concat2(conv2, up3)#
        #print(up2.shape)2, 32, 48, 48, 48
        up1 = self.up_concat1(conv1, up2)#
        #print(up1.shape) 2, 16, 96, 96, 96
        up1 = self.dropout2(up1)
        
        final = self.final(up1)
        feature_rep = self.representation(up1)
        mm_feathre=[up3,up2]
        if mm:
            feat_Map,prototype= self.dynamic(mm_feathre)
            return final, feature_rep,feat_Map,prototype
        else:
            return final,feature_rep

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class Dynamicinteraction(nn.Module):
    def __init__(
            self,
            in_chans: Sequence[int],
            depths: Sequence[int],
            input_resolution: Sequence[int],
            num_classes: int,
            num_pro: int,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            patch_norm: bool = False,
            spatial_dims: int = 3,
            drop_path_rate: float = 0.1

    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.patch_norm = patch_norm
        self.depth = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.proj_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.pairwise_interaction = nn.ModuleList()
        self.attn_convs0 = nn.ModuleList()
        self.attn_convs1 = nn.ModuleList()
        self.query_convs = nn.ModuleList()

        for i_layer in range(len(depths)):
            self.proj_layers.append(
                Conv[Conv.CONV, spatial_dims](in_channels=in_chans[i_layer], out_channels=in_chans[i_layer],
                                              kernel_size=(1, 1, 1), stride=(1, 1, 1)))
            self.norm_layers.append(norm_layer(in_chans[i_layer]))
            self.pairwise_interaction.append(
                Pairwise_interaction(dim=in_chans[i_layer], input_resolution=input_resolution[i_layer],
                                     mlp_ratio=4.,
                                     qkv_bias=True, qk_scale=None,
                                     drop=0., attn_drop=0.,
                                     drop_path=dpr[1],
                                     norm_layer=norm_layer))
            self.attn_convs0.append(
                SeparableConv3d(num_pro, num_pro, (3, 3, 3), norm_layer=nn.BatchNorm3d,
                                relu_first=False))
            self.attn_convs1.append(
                nn.Conv3d(in_channels=num_pro, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1)))

            self.query_convs.append(
                nn.Conv2d(in_channels=in_chans[i_layer], out_channels=32, kernel_size=1, stride=1,
                          padding=0))

        self.class_prototype = nn.Parameter(torch.zeros(1, num_classes, num_pro, in_chans[0]))
        self.block_1 = UpsamplingDeconvBlock(2, 2)

    def forward(self, feats):
        BS = feats[0].shape[0]

        update_class_prototype = self.class_prototype.expand(BS, -1, -1, -1)

        for i_layer in range(len(self.depth)):
            tok_feats = self.norm_layers[i_layer](
                self.proj_layers[i_layer](feats[i_layer]).flatten(2).transpose(1, 2))
            update_class_prototype, attn_map = self.pairwise_interaction[i_layer](update_class_prototype, tok_feats)

            bs, num_classes, num_pro, _ = attn_map.size()
            d, h, w = feats[i_layer].shape[2], feats[i_layer].shape[3], feats[i_layer].shape[4]

            attn_map = attn_map.contiguous().view(bs, num_classes, num_pro, d, h, w)
            attn_map = attn_map.reshape(bs * num_classes, num_pro, d, h, w)
            attn_map = self.attn_convs0[i_layer](attn_map)
            rectification_cue = self.attn_convs1[i_layer](attn_map).squeeze(1).reshape(bs, num_classes, d, h, w)
            update_class_prototype = self.query_convs[i_layer](update_class_prototype.permute(0, 3, 1, 2)).squeeze(1)
            update_class_prototype = update_class_prototype.permute(0, 2, 3, 1)

        rectification_cue = self.block_1(rectification_cue)

        return rectification_cue, self.class_prototype


class Pairwise_interaction(nn.Module):
    def __init__(self, dim, input_resolution, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_query = norm_layer(dim)
        self.attn = Pair_Attention(
            dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(input_resolution[0] * input_resolution[1] * input_resolution[2])

    def forward(self, query, feat):
        query, attn = self.attn(self.norm1_query(query), self.norm1(feat))
        query = query + self.drop_path(query)
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        attn = attn + self.drop_path(attn)
        attn = attn + self.drop_path(self.norm3(attn))
        return query, attn


class Pair_Attention(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5
        self.fc_q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.fc_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, x):
        B, N, C = x.shape
        num_classes = q.shape[1]
        num_proto_set = q.shape[2]
        q = self.fc_q(q).reshape(B, num_proto_set, num_classes, C)
        kv = self.fc_kv(x).reshape(B, N, 2, 1, C).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn1 = (q.matmul(k.transpose(-2, -1)) * self.scale)
        attn2 = attn1.softmax(dim=-1)
        attn3 = self.attn_drop(attn2)
        x = (attn3.matmul(v)).reshape(B, num_proto_set, num_classes, C).permute(0, 2, 1, 3)
        x = self.proj(x)
        x = self.proj_drop(x)
        attn = attn1.permute(0, 2, 1, 3)

        return x, attn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1), relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm3d):
        super().__init__()
        depthwise = nn.Conv3d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv3d(inplanes, planes, (1, 1, 1), bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))

    def forward(self, x):
        return self.block(x)