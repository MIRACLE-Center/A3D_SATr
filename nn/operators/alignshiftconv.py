import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
import numpy as np
from ..utiles import as_triple, _pair_same


class AlignShiftConv(_ConvNd):
    '''
    Args: 
        n_fold (int): Divisor of channels to shift.
        alignshift(bool): if apply alignshift operation before conv
        inplace (bool): if Enable inplace operation.
        ref_thickness (float): Reference z axis spacing Default: 0.2mm.
        shift_padding_zero(bool): f padding zeros to side fold before shift channels 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', 
                 n_fold=8, alignshift=True, inplace=True,
                 ref_thickness=2., shift_padding_zero=False):

        kernel_size = _pair_same(kernel_size)
        stride = as_triple(stride)
        padding = as_triple(padding, 0)
        dilation = as_triple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, as_triple(0,0), groups, bias, padding_mode)
        self.n_fold = n_fold
        self.enable_align = alignshift
        self.inplace = inplace
        self.ref_thickness = ref_thickness
        self.shift_padding_zero = shift_padding_zero

    def alignshift(self, input, fold, ref_thickness, thickness, padding_zero=True):
        alph = 1 - (ref_thickness / thickness).view(-1, 1 ,1, 1, 1).clamp_(0., 1.)
        out = torch.zeros_like(input)
        out[:, :fold, :-1] = input[:,  :fold, :-1] * alph +  input[:,  :fold, 1:] * (1-alph)
        out[:, fold: 2 * fold, 1:] = input[:,  fold: 2 * fold, 1:] * alph + \
                                             input[:,  fold: 2 * fold, :-1] * (1-alph)
        out[:, 2 * fold:, :] = input[: , 2 * fold:, :]
        pad_alph = alph if padding_zero else 1.0
        out[:, :fold, -1:] = pad_alph * input[:,  :fold, -1:]
        out[:, fold:2*fold, :1] = pad_alph * input[:, fold:2*fold, :1]        
        return out

    def align_shift(self, x, fold, ref_thickness, thickness, padding_zero, inplace):
        if inplace:
            x = inplace_alignshift(x, fold, ref_thickness, thickness, padding_zero)
        else:
            x = self.alignshift(x, fold, ref_thickness, thickness, padding_zero)
        return x

    def forward(self, input, thickness=None):
        if self.enable_align:
            _, c, _, _, _ = input.size()
            fold = c // self.n_fold 
            input = self.align_shift(input, fold, self.ref_thickness, \
                thickness, self.shift_padding_zero, self.inplace)
        return F.conv3d(input, self.weight.unsqueeze(2), self.bias, self.stride,
                        self.padding, self.dilation, self.groups) 

    def extra_repr(self):
        s = super().extra_repr() + ', alignshift={self.alignshift}'
        return s.format(**self.__dict__)

class InplaceAlignShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fold, ref_thickness, thickness, padding_zero=True):
        '''
        @params: 
            input: BxCxDxHxW
            fold: channels to align
            align_spacing: align spacing
            thickness: thickness of the input ct
            padding_zero: bool, whether padding zero to outer cts 
        '''
        n, c, t, h, w = input.size()
        ctx.fold_ = fold
        ctx.padding_zero = padding_zero
        alph = 1 - (ref_thickness / thickness).view(-1, 1 ,1, 1, 1).clamp_(0., 1.) ##把小于ref_thickness的当作ref_thickness，不做插值处理 
        ctx.alph_ = alph
        input.data[:, :fold, :-1] = input.data[:, :fold, :-1] * alph + input.data[:, :fold, 1:] * (1-alph)
        input.data[:, fold:2*fold, 1:] = input.data[:, fold:2*fold, 1:] * alph + \
                                                                input.data[:, fold:2*fold, :-1] * (1-alph)
        if padding_zero:
            input.data[:, :fold, -1:] =  input.data[:, :fold, -1:] * alph
            input.data[:, fold:2*fold, :1] = input.data[:, fold:2*fold, :1] * alph
            # input.data[:, :fold, -1:] =  input.data[:, :fold, -1:] * 0.6
            # input.data[:, fold:2*fold, :1] = input.data[:, fold:2*fold, :1] * 0.6
        return input

    @staticmethod
    def backward(ctx, grad_output):
        fold = ctx.fold_
        alph = ctx.alph_
        padding_zero = ctx.padding_zero
        n, c, t, h, w = grad_output.size()
        pad_fold = alph if padding_zero else 1.      
        grad_output.data[:, :fold, -1:] = grad_output.data[:, :fold, -1:] \
                                                * pad_fold + grad_output.data[:, :fold, -2:-1] * (1-alph)
        grad_output.data[:, :fold, 1:-1] = grad_output.data[:, :fold, 1:-1] * alph\
                                                             + grad_output.data[:, :fold, :-2] * (1-alph)
        grad_output.data[:, :fold, :1] = grad_output.data[:, :fold, :1] * alph       
        grad_output.data[:, fold:2*fold, :1] = grad_output.data[:, fold:2*fold, :1] * pad_fold\
                                                        + grad_output.data[:, fold:2*fold, 1:2] * (1-alph)                                            
        grad_output.data[:, fold:2*fold, 1:-1] = grad_output.data[:, fold:2*fold, 1:-1] * alph\
                                                        + grad_output.data[:, fold:2*fold, 2:] * (1-alph)                                    
        grad_output.data[:, fold:2*fold, -1:] = grad_output.data[:, fold:2*fold, -1:] * alph
        return grad_output, None, None, None, None

inplace_alignshift = InplaceAlignShift.apply

