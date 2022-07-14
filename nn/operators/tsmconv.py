import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
import numpy as np
from ..utiles import as_triple, _pair_same


class TSMConv(_ConvNd):
    '''
    Args: 
        n_fold (int): Divisor of channels to shift.
        tsm(bool) : if apply tsm operation before conv
        inplace (bool): if Enable inplace opertion.
        shift_padding_zero(bool): if padding zeros to side fold before shift channels 

    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', 
                 n_fold=8, tsm=True, inplace=False,
                 shift_padding_zero=True):

        kernel_size = _pair_same(kernel_size)
        stride = as_triple(stride)
        padding = as_triple(padding, d_value=0)
        dilation = as_triple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, as_triple(0,0), groups, bias, padding_mode)
        self.n_fold = n_fold
        self.enable_tsm = tsm
        self.inplace = inplace
        self.shift_padding_zero = shift_padding_zero

    def tsm(self, input, fold):
        out = torch.zeros_like(input)
        out[:, :fold, :-1] = input[:,  :fold, 1:]
        out[:, fold: 2 * fold, 1:] = input[:,  fold: 2 * fold, :-1]
        out[:, 2 * fold:, :] = input[: , 2 * fold:, :]       
        return out

    def tsm_shift(self, x, fold, inplace):
        if inplace:
            x = inplace_tsm(x, fold)
        else:
            x = self.tsm(x, fold)
        return x

    def forward(self, input):
        if self.enable_tsm:
            _, c, _, _, _ = input.size()
            fold = c // self.n_fold     
            input = self.tsm_shift(input, fold, self.inplace)
        return F.conv3d(input, self.weight.unsqueeze(2), self.bias, self.stride,
                        self.padding, self.dilation, self.groups) 

    def extra_repr(self):
        s = super().extra_repr() + ', tsm={self.tsm}'
        return s.format(**self.__dict__)


class InplaceTSM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fold):
        '''
        @params: 
            input: BxCxDxHxW
            fold: chennels of shift part
        '''
        # not support higher order gradient
        n, c, t, h, w = input.size()
        ctx.fold_ = fold
        buffer = input.data.new(n, fold, t, h, w).zero_()
        buffer[:, :, :-1] = input.data[:, :fold, 1:]
        input.data[:, :fold, :] = buffer
        buffer.zero_()
        buffer[:, :, 1:] = input.data[:, fold: 2 * fold, :-1]
        input.data[:, fold: 2 * fold, :] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        fold = ctx.fold_
        n, c, t, h, w = grad_output.size()
        buffer = grad_output.data.new(n, fold, t, h, w).zero_()
        buffer[:, :, 1:] = grad_output.data[:, :fold, :-1]
        grad_output.data[:, :fold, :] = buffer
        buffer.zero_()
        buffer[:, :, :-1] = grad_output.data[:, fold: 2 * fold, 1:]
        grad_output.data[:, fold: 2 * fold, :] = buffer
        return grad_output, None


inplace_tsm = InplaceTSM.apply

