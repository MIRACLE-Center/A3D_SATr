# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from torchvision.models.densenet import model_urls
import math
import torch.utils.model_zoo as model_zoo
import re
from collections import OrderedDict
from mmdet.models.registry import BACKBONES
from nn.operators import AlignShiftConv
import torch.utils.checkpoint as cp
from mmdet.models.utils import build_conv_layer, build_norm_layer
# mybn = nn.BatchNorm3d
# mybn = nn.SyncBatchNorm
norm_cfg = dict(type='SyncBN')
def _bn_function_factory(norm, relu, conv, thickness):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)), thickness)
        return bottleneck_output
    return bn_function

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, n_fold, memory_efficient=False, ref_thickness=None):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', build_norm_layer(norm_cfg, num_input_features, postfix=1)[1]),
        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', AlignShiftConv(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, alignshift=True,
                                           bias=False, ref_thickness=ref_thickness, n_fold=n_fold,)),
        self.add_module('norm2', build_norm_layer(norm_cfg, bn_size * growth_rate, postfix=1)[1]),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', AlignShiftConv(bn_size * growth_rate, growth_rate, alignshift=True,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False, ref_thickness=ref_thickness, n_fold=n_fold)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features, thickness):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1, thickness=thickness)
        if self.memory_efficient:# and any(prev_feature.requires_grad for prev_feature in prev_features):hyadd 这里自己设计节省效率
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features, thickness=thickness)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)), thickness)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, n_fold, memory_efficient=False, ref_thickness=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                ref_thickness=ref_thickness,
                n_fold=n_fold
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features, thickness):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features, thickness=thickness)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', build_norm_layer(norm_cfg, num_input_features, postfix=1)[1])
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', AlignShiftConv(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False, alignshift=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2]))


class _Reduction_z(nn.Sequential):
    def __init__(self, input_features, input_slice):
        super().__init__()
        self.add_module('reduction_z_conv', nn.Conv3d(input_features, input_features, kernel_size=[input_slice, 1, 1],
                                                    stride=1, bias=False))
        # self.add_module('reduction_z_pooling', nn.AvgPool3d(kernel_size=[input_slice, 1, 1], stride=1))
@BACKBONES.register_module
class DenseNetCustomTrunc3dAlign(nn.Module):
    def __init__(self, 
                out_dim=256,
                n_cts=3,
                fpn_finest_layer=1,
                ref_thickness=2.0,
                n_fold=8, 
                memory_efficient=True):
        super().__init__()
        self.depth = 121
        self.feature_upsample = True
        self.fpn_finest_layer = fpn_finest_layer
        self.out_dim = out_dim
        self.n_cts = n_cts
        self.mid_ct = n_cts//2
        self.ref_thickness = ref_thickness
        self.n_fold = n_fold
        assert self.depth in [121]
        if self.depth == 121:
            num_init_features = 64
            growth_rate = 32
            block_config = (6, 12, 24)
            self.in_dim = [64, 256, 512, 1024]
        bn_size = 4
        drop_rate = 0

        # First convolution
        self.conv0 = AlignShiftConv(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False, alignshift=False)
        self.norm0 = build_norm_layer(norm_cfg, num_init_features, postfix=1)[1]
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, ref_thickness=self.ref_thickness,
                                n_fold=self.n_fold, memory_efficient=memory_efficient)
            self.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            reductionz = _Reduction_z(num_features, self.n_cts)
            # normrelu = _StageNormRelu(num_features)
            # self.add_module('normrelu%d' % (i + 1), normrelu)
            self.add_module('reductionz%d' % (i + 1), reductionz)
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2


        # Final batch norm
        # self.add_module('norm5', nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if self.feature_upsample:
            for p in range(4, self.fpn_finest_layer - 1, -1):
                layer = nn.Conv2d(self.in_dim[p - 1], self.out_dim, 1)
                name = 'lateral%d' % p
                self.add_module(name, layer)

                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)
        self.init_weights()
        # if syncbn:
        #     self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward(self, x, thickness):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        x = self.denseblock1(x, thickness)
        # x = self.normrelu1(x)
        redc1 = self.reductionz1(x)
        x = self.transition1(x)


        x = self.denseblock2(x, thickness)
        # x = self.normrelu2(x)
        redc2 = self.reductionz2(x)
        x = self.transition2(x)


        x = self.denseblock3(x, thickness)
        # x = self.normrelu3(x)
        redc3 = self.reductionz3(x)
        # truncated since here since we find it works better in DeepLesion
        # ts3 = self.transition3(db3)
        # db4 = self.denseblock4(ts3)

        # if self.feature_upsample:
        ftmaps = [None, redc1.squeeze(2), redc2.squeeze(2), redc3.squeeze(2)]
        x = self.lateral4(ftmaps[-1])
        for p in range(3, self.fpn_finest_layer - 1, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            y = ftmaps[p-1]
            lateral = getattr(self, 'lateral%d' % p)(y)
            x += lateral
        return [x]
        # else:
        #     return [db3]

    def init_weights(self, pretrained=True):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        state_dict1 = {}
        for key in list(state_dict.keys()):
            new_key = key.replace('features.', '')
            if state_dict[key].dim() == 4:           
                t0 = state_dict[key].shape[1]
                state_dict1[new_key] = state_dict[key]#.unsqueeze(2)#.repeat((1,1,self.n_cts,1,1))/self.n_cts
                if t0 == 3:
                    state_dict1[new_key] = state_dict1[new_key][:,1:2,...]
            else:
                state_dict1[new_key] = state_dict[key]

        key = self.load_state_dict(state_dict1, strict=False)
        #print(key)
        
    def freeze(self):
        for name, param in self.named_parameters():
            print('freezing', name)
            param.requires_grad = False

