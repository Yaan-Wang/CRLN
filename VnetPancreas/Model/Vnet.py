import torch.nn
from torch import nn
from monai.networks.layers import DropPath, Conv
from monai.utils import ensure_tuple_rep
from collections import OrderedDict
import numpy as np
from typing import Sequence, Type
from torch.nn import LayerNorm


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


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


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()
        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
class opmoudle(nn.Module):
    def __init__(self):
      super(opmoudle, self).__init__()

      self.a=nn.Parameter(torch.zeros(1))

    def forward(self, or_pre,mm_pre):
        op_pre=or_pre+(1-self.a)*mm_pre

        return op_pre


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, mun_pro=16,normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        self.mun_pro=mun_pro

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.mlp_proj = MlpProjector(n_channels=n_filters * 8)
        self.representation = nn.Sequential(
            nn.Conv3d(16, self.mun_pro, 3, padding=1, bias=False),
            nn.BatchNorm3d(self.mun_pro),
            nn.ReLU(),
            nn.Conv3d(self.mun_pro, self.mun_pro, 1)
        )

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

        self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
     
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        # 4, 16, 112, 112, 80
        x1 = features[0]

        # 4, 32, 56, 56, 40
        x2 = features[1]

        # 4, 64, 28, 28, 20
        x3 = features[2]

        # 4, 128, 14, 14, 10
        x4 = features[3]

        # 4, 256, 7, 7, 5
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv(x9)
        feature_rep=self.representation(x9)
        return out, feature_rep,[x7,x8]
    
    def forward(self, input, mm=False,turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout 
            self.has_dropout = False

        features = self.encoder(input)
        out, feature_rep,mm_feathre = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        if mm:
            feat_Map,prototype= self.dynamic(mm_feathre)
            return out, feature_rep,feat_Map,prototype
        else:
            return out,feature_rep


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MlpProjector(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv3d(in_channels=n_channels, out_channels=n_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm3d(n_channels),
            nn.ReLU(inplace=True)
        )
        self.layer2 = torch.nn.Sequential(
            nn.Conv3d(n_channels, n_channels * 2,
                      1, stride=2, padding=0),
            nn.BatchNorm3d(n_channels * 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Dynamicinteraction(nn.Module):
    def __init__(
            self,
            in_chans: Sequence[int], 
            depths: Sequence[int],
            input_resolution: Sequence[int],
            num_classes: int,
            num_pro:int,
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

        self.class_prototype = nn.Parameter(torch.zeros(1, num_classes, num_pro,in_chans[0]))
        self.block_1 = UpsamplingDeconvBlock(2, 2)

    def forward(self, feats):
        BS = feats[0].shape[0]

        update_class_prototype = self.class_prototype.expand(BS, -1, -1,-1)

        for i_layer in range(len(self.depth)):

            tok_feats = self.norm_layers[i_layer](
                self.proj_layers[i_layer](feats[i_layer]).flatten(2).transpose(1, 2))
            update_class_prototype, attn_map = self.pairwise_interaction[i_layer](update_class_prototype, tok_feats)
    
            bs, num_classes, num_pro, _ = attn_map.size()
            d,h,w=feats[i_layer].shape[2],feats[i_layer].shape[3],feats[i_layer].shape[4]
          
            attn_map = attn_map.contiguous().view(bs, num_classes, num_pro,d, h, w)
            attn_map = attn_map.reshape(bs * num_classes, num_pro, d, h, w)
            attn_map = self.attn_convs0[i_layer](attn_map)
            rectification_cue= self.attn_convs1[i_layer](attn_map).squeeze(1).reshape(bs, num_classes, d, h, w)
            update_class_prototype = self.query_convs[i_layer](update_class_prototype.permute(0, 3, 1,2)).squeeze(1)
            update_class_prototype = update_class_prototype.permute(0, 2, 3, 1)

        rectification_cue=self.block_1(rectification_cue)
      
        
        return rectification_cue,self.class_prototype 


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
        num_proto_set=q.shape[2]
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