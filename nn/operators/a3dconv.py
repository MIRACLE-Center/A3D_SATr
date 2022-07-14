import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from ..utiles import as_triple, _pair_same
from torch.nn.parameter import Parameter


class A3DConv(_ConvNd):
    '''
    Args: 
        dimension (int): number of dimensions to shift.
        inplace (bool): if Enable inplace operation.
        enable_shift(bool): if enable shift 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', 
                 dimension=3, enable_shift=True, inplace=False):

        kernel_size = _pair_same(kernel_size)
        stride = as_triple(stride)
        padding = as_triple(padding, 0)
        dilation = as_triple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, as_triple(0,0), groups, bias, padding_mode)

        self.enable_shift = enable_shift
        self.inplace = inplace
        self._init_adaptive_weights(in_channels, dimension)

    def _init_adaptive_weights(self, in_channels, dimension):
        self.adaptive_align_weights = Parameter(torch.Tensor(dimension, dimension, in_channels))
        with torch.no_grad():
            aw = torch.eye(dimension).unsqueeze(-1).repeat((1,1,in_channels))
            noise = aw.new(aw.shape)
            init.uniform_(noise, -0.1, 0.1)
            self.adaptive_align_weights.data = aw + noise
        # self.adaptive_align_weights.requires_grad = True

    def _adaptive_shift(self, input):
        out = torch.einsum('bcdhw,dkc->bckhw', [input, self.adaptive_align_weights])
        return out

    def adaptive_shift(self, x, inplace):
        if inplace:
            raise NotImplementedError
        else:
            x = self._adaptive_shift(x)
        return x

    def forward(self, input):
        if self.enable_shift:
            _, c, _, _, _ = input.size()
            input = self.adaptive_shift(input, self.inplace)
        return F.conv3d(input, self.weight.unsqueeze(2), self.bias, self.stride,
                        self.padding, self.dilation, self.groups) 

    def extra_repr(self):
        s = super().extra_repr() + f', adaptive_shift={self.enable_shift}'
        return s.format(**self.__dict__)

    def flops(self, input, output):
        
        batch_size = input.shape[0]
        output_dims = list(output.shape[2:])

        kernel_dims = list(self.kernel_size)
        in_channels = self.in_channels
        out_channels = self.out_channels
        groups = self.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = np.prod(
            kernel_dims) * in_channels * filters_per_channel
        active_elements_count = batch_size * np.prod(output_dims)

        if self.__mask__ is not None:
            # (b, 1, h, w)
            output_height, output_width = output.shape[2:]
            flops_mask = self.__mask__.expand(batch_size, 1, output_height,
                                                    output_width)
            active_elements_count = flops_mask.sum()

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0

        if self.bias is not None:

            bias_flops = out_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops 

        # adaptive_weights_flops
        if self.enable_shift:
            adaptive_align_weights_flops = np.prod(self.adaptive_align_weights.shape)# * in_channels ** 2
            overall_flops += adaptive_align_weights_flops
        #
        return overall_flops  