import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn.weight_init import caffe2_xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
import torch
import gc


@NECKS.register_module
class FPNLast(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            #assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            raise 'please dont.'
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(1):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        # assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')


        # build outputs
        # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]
        for l in self.fpn_convs:
            laterals[0] = l(laterals[0])

        return tuple([laterals[0]])


@NECKS.register_module
class FPNCatLast(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            #assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            raise 'please dont.'
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        fpn_conv = ConvModule(
            out_channels*(len(self.lateral_convs)-1),
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=self.activation,
            inplace=False)

        self.fpn_convs.append(fpn_conv)


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        out = laterals[-1]
        for i in range(used_backbone_levels - 2, 0, -1):
            out = torch.cat([laterals[i], F.interpolate(out, scale_factor=2, mode='nearest')], dim=1)

        # build outputs
        # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]

        return tuple([self.fpn_convs[0](out)])



@NECKS.register_module
class HRFPNLast(nn.Module):
    """HRFPN (High Resolution Feature Pyrmamids)

    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        stride (int): stride of 3x3 convolutional layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=1,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False,
                 stride=1):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.reduction_conv = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            activation=None)

        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    activation=None))

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        outs = inputs[0]
        for i in range(1, self.num_ins):
            del inputs[0]
            gc.collect()
            outs = torch.cat([outs, F.interpolate(inputs[0], scale_factor=2**i, mode='bilinear')], dim=1)
        if outs.requires_grad and self.with_cp:
            outs = checkpoint(self.reduction_conv, outs)
        else:
            outs = self.reduction_conv(outs)
        # outs = [out]
        # for i in range(1, self.num_outs):
        #     outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        # outputs = []

        # for i in range(self.num_outs):
        #     if outs[i].requires_grad and self.with_cp:
        #         tmp_out = checkpoint(self.fpn_convs[i], outs[i])
        #     else:
        #         tmp_out = self.fpn_convs[i](outs[i])
        #     outputs.append(tmp_out)
        
        return tuple([self.fpn_convs[0](outs)])

@NECKS.register_module
class MyFPN(nn.Module):
    def  __init__(self, in_channels, out_channels):
        super().__init__()
        for p in range(4, 0, -1):
            layer = nn.Conv2d(in_channels[p - 1], out_channels, 1)
            name = 'lateral%d' % p
            self.add_module(name, layer)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x = self.lateral4(inputs[-1])
        for p in range(3, 0, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            lateral = getattr(self, 'lateral%d' % p)(inputs[p-1])
            x += lateral
        
        return tuple([x])

@NECKS.register_module
class MyFPN3(nn.Module):
    def  __init__(self, in_channels, out_channels):
        super().__init__()
        for p in range(3, 0, -1):
            layer = nn.Conv2d(in_channels[p - 1], out_channels, 1)
            name = 'lateral%d' % p
            self.add_module(name, layer)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, inputs):
        x = self.lateral3(inputs[-2])
        for p in range(2, 0, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            lateral = getattr(self, 'lateral%d' % p)(inputs[p-1])
            x += lateral
        
        return tuple([x])