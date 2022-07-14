# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from torchvision.models.densenet import _DenseBlock, _Transition, model_urls
import math
import torch.utils.model_zoo as model_zoo
import re
from collections import OrderedDict
from mmdet.models.registry import BACKBONES

@BACKBONES.register_module
class DenseNetCustomTrunc(nn.Module):
    def __init__(self, 
                out_dim=256,
                in_channels=3,
                fpn_finest_layer=1):
        super().__init__()
        self.depth = 121
        self.feature_upsample = True
        self.fpn_finest_layer = fpn_finest_layer
        self.out_dim = out_dim
        self.in_channel = in_channels
        assert self.depth in [121]
        if self.depth == 121:
            num_init_features = 64
            growth_rate = 32
            block_config = (6, 12, 24)
            self.in_dim = [64, 256, 512, 1024]
        bn_size = 4
        drop_rate = 0

        # First convolution
        self.conv0 = nn.Conv2d(self.in_channel, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, \
                                    memory_efficient=True)
            self.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        # self.add_module('norm5', nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
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

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        relu0 = self.relu0(x)
        pool0 = self.pool0(relu0)

        db1 = self.denseblock1(pool0)
        ts1 = self.transition1(db1)

        db2 = self.denseblock2(ts1)
        ts2 = self.transition2(db2)

        db3 = self.denseblock3(ts2)

        # truncated since here since we find it works better in DeepLesion
        # ts3 = self.transition3(db3)
        # db4 = self.denseblock4(ts3)

        if self.feature_upsample:
            ftmaps = [relu0, db1, db2, db3]
            x = self.lateral4(ftmaps[-1])
            for p in range(3, self.fpn_finest_layer - 1, -1):
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                y = ftmaps[p-1]
                lateral = getattr(self, 'lateral%d' % p)(y)
                x += lateral
            return [x]
        else:
            return [db3]

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
            if self.in_channel != 3:
                if len(state_dict[key].shape)==4 and state_dict[key].shape[1]==3:
                    a = self.in_channel // 3 + 1
                    b = self.in_channel % 3 + 1                     
                    state_dict[key] = state_dict[key].repeat((1,a,1,1))[:,b//2:b//2-b,...]
            state_dict1[new_key] = state_dict[key]
        self.load_state_dict(state_dict1, strict=False)

    def freeze(self):
        for name, param in self.named_parameters():
            print('freezing', name)
            param.requires_grad = False