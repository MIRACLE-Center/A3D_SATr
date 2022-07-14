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
from nn.operators import A3DConv
import torch.utils.checkpoint as cp
from mmdet.models.utils import build_conv_layer, build_norm_layer
from deeplesion.mconfigs.densenet_a3d import backbone_manner,RPN_head_decoupled,Unet_FPN_default,Fusion_manner,Trans_thick_aware,Sep_pre_Conv
from functools import partial
import numpy as np

# mybn = nn.BatchNorm3d
# mybn = nn.SyncBatchNorm
norm_cfg = dict(type='SyncBN')
# norm_cfg = dict(type='BN')
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

from torch import Tensor
import torch.nn.functional as F

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
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

class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x

class simple_ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(simple_ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(outplanes)
        self.act1 = act_layer(inplace=True)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t):
        residual = x

        x = self.conv1(x + x_t)
        x = self.bn1(x)
        x = self.act1(x)
        return (x+residual).unsqueeze(2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.qk_conv=nn.Conv3d()
        if Fusion_manner==0 or Fusion_manner==1:
            self.qk = nn.Linear(dim*6, dim * 2, bias=qkv_bias)#han_add
        elif Fusion_manner==2 or Fusion_manner==3 :
            self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)  # han_add
        self.v = nn.Linear(dim, dim, bias=qkv_bias)#han_add
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x):#han_add
        B, S,N, C = x.shape# B batch, S: num of slice,N, dimention +1 C channel 384
        if Fusion_manner==0 or Fusion_manner==1:
            mid=int(S//2)
            v_slice=x[:,mid,:,:]
            id_list=list(np.arange(S))
            id_list.remove(mid)
            qk_slice=x.index_select(1,torch.tensor(id_list).cuda())
        elif Fusion_manner==2 or Fusion_manner==3:
            v_slice=x[:,1,:,:]
            qk_slice=x[:,0,:,:].unsqueeze(1)

        qk=self.qk(qk_slice.permute(0, 2, 1, 3).reshape(B, N, C * (S-1)))
        qk=qk.reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v=self.v(v_slice).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #qkv: B, N, 3*C;       reshape: B,N,3(qkv),heads,C//heads;         permute: 3(qkv), B,heads,N, C//heads
        q, k, v= qk[0], qk[1], v[0]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale#q*k *scale, no mask out here
        #k(B,heads,N,C//heads)--transpose-->k(B,heads,C//heads,N)//// q*k--->B,heads,N,N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)# the heads has been removed by concating the heads
        #(attn B,heads,N,N) * v (B,heads,N,C//heads)---->B,heads,N,C//heads-----trans--->B,N,heads,C//heads---reshape-->B,N,C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if Fusion_manner==0 or Fusion_manner==1:
            mid_x=x[:,3,:,:]
            # x = x + self.drop_path(self.attn(self.norm1(x)))
            x = mid_x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        elif Fusion_manner == 2:
            mid_x = x[:,1,:,:]
            x = mid_x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class double_conv(nn.Module):##两次卷积
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        ##conv的初始化
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            ##直接写了一个con的顺序
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            ##归一化
            nn.ReLU(inplace=True),##inplace就是要不要修改原对象
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            ##3 是kenel_size,他可以写成变的，（3,5） pading也可以 （1,2）

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:##当一个开关
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
                ##用的是反卷积
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        ##这里是拼接的过程，
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        ##找到差值，进行尺寸修正
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        # self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        if Fusion_manner==0 or Fusion_manner==1: #concat feature from slices
            for i in range(7):

                self.add_module('conv_project%d' % (i + 1), nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0))
                self.add_module('sample_pooling%d'% (i + 1),nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride))
                self.add_module('ln%d'% (i + 1),norm_layer(outplanes))
                self.add_module('act%d'% (i + 1),act_layer())
            # self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
            # self.ln = norm_layer(outplanes)
            # self.act = act_layer()
        elif Fusion_manner == 2:
            self.conv_project3D=nn.Conv3d(inplanes, outplanes,kernel_size=(6,1,1),stride=(6,1,1),padding=0)
            self.conv_project2D=nn.Conv2d(inplanes, outplanes,kernel_size=(1,1),stride=(1,1),padding=0)

            self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
            self.ln3D = norm_layer(outplanes)
            self.ln2D = norm_layer(outplanes)
            self.act = act_layer()
        elif Fusion_manner == 3:
            self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
            self.ln = norm_layer(outplanes)
            self.act = act_layer()
    def forward(self, x, x_t,thicks=None):
        ref_thick=2
        B,C,N,W,H=x.shape
        if thicks is None:
            thicks=[2]*B
        tem=[]
        mid=int(N//2)
        thick_ratio=torch.ones_like(x[:,:,:,:,:])
        if Trans_thick_aware:
            for i,thick in enumerate(thicks):
                if thick>ref_thick:
                    # min_r=ref_thick/thick
                    min_r=0.5
                    for n in range(N):
                        if n !=mid:
                            r=(3-abs(mid-n))*0.25+min_r
                            thick_ratio[i,:,n,:,:]=r

        if Fusion_manner==0 or Fusion_manner==1:
            for n in range(N):
                # tem_x = self.conv_project(x[:,:,n,:,:])  # [N, C, H, W]
                tem_x = getattr(self,'conv_project%d'%(n+1))(x[:,:,n,:,:]*thick_ratio[:,:,n,:,:])  # [N, C, H, W]
                tem_x = getattr(self,'sample_pooling%d' %(n+1))(tem_x).flatten(2).transpose(1, 2)
                tem_x = getattr(self,'ln%d'%(n+1))(tem_x)
                tem_x = getattr(self,'act%d' %(n+1))(tem_x)
                tem_x = torch.cat([x_t[:, 0][:, None, :], tem_x], dim=1)
                if n==mid:# the mid slice will add all infor from last trans layer
                    tem_x+=x_t
                tem.append(tem_x)
            return torch.stack(tem,dim=1)
        elif Fusion_manner==2:
            # thick_ratio = torch.ones_like(x[:, :, :, :, :])
            mid_slice=x[:,:,mid,:,:]
            adjacent_slice=torch.cat((x[:,:,:3,:,:],x[:,:,4:,:,:]),dim=2)
            mid_f=self.conv_project2D(mid_slice)
            mid_f=self.sample_pooling(mid_f).flatten(2).transpose(1, 2)
            mid_f=self.ln2D(mid_f)
            mid_f=self.act(mid_f)
            mid_f = torch.cat([x_t[:, 0][:, None, :], mid_f], dim=1)
            mid_f+=x_t
            adjacent_f=self.conv_project3D(adjacent_slice)
            adjacent_f=adjacent_f.squeeze(dim=2)
            adjacent_f=self.sample_pooling(adjacent_f).flatten(2).transpose(1, 2)
            adjacent_f=self.ln3D(adjacent_f)
            adjacent_f=self.act(adjacent_f)
            adjacent_f = torch.cat([x_t[:, 0][:, None, :], adjacent_f], dim=1)
            return torch.stack([adjacent_f,mid_f],dim=1)





class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes,  act_layer=nn.ReLU,up_stride=8,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H*self.up_stride, W*self.up_stride))


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t,thicks):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t,thicks)

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class Sep_2D_Conv(nn.Module):
    def __init__(self,input_features,output_features,kernel_size,stride,padding=0,num_slices=7):
        super().__init__()
        self.num_slices=num_slices
        for i in range(self.num_slices):
            self.add_module('conv_%d' % (i + 1), nn.Conv2d(input_features, output_features, kernel_size=kernel_size,stride=stride,padding=padding))

    def forward(self, x):
        tem=[]
        for i in range(self.num_slices):
            tem_x= getattr(self,'conv_%d'%(i+1))(x[:, :, i, :, :])
            tem.append(tem_x)
        return torch.stack(tem, dim=1)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dimension, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', build_norm_layer(norm_cfg, num_input_features, postfix=1)[1]),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', A3DConv(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, enable_shift=True,
                                           dimension=dimension, bias=False)),
        self.add_module('norm2', build_norm_layer(norm_cfg, bn_size * growth_rate, postfix=1)[1]),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', A3DConv(bn_size * growth_rate, growth_rate, enable_shift=True,
                                           kernel_size=3, stride=1, padding=1,
                                           dimension=dimension, bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient:# and any(prev_feature.requires_grad for prev_feature in prev_features):hyadd 这里自己设计节省效率
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dimension=None, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                dimension=dimension,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', build_norm_layer(norm_cfg, num_input_features, postfix=1)[1])
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', A3DConv(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False, enable_shift=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2]))


class _Reduction_z(nn.Sequential):
    def __init__(self, input_features, input_slice):
        super().__init__()
        self.add_module('reduction_z_conv', nn.Conv3d(input_features, input_features, kernel_size=[input_slice, 1, 1],
                                                    stride=1, bias=False))
        # self.add_module('reduction_z_pooling', nn.AvgPool3d(kernel_size=[input_slice, 1, 1], stride=1))
@BACKBONES.register_module
class DenseNetCustomTrunc3dA3D(nn.Module):
    def __init__(self, 
                out_dim=256,
                n_cts=3,
                fpn_finest_layer=1,
                memory_efficient=True):
        super().__init__()
        self.depth = 121
        self.feature_upsample = True
        self.fpn_finest_layer = fpn_finest_layer
        self.out_dim = out_dim
        self.n_cts = n_cts
        self.mid_ct = n_cts//2

        assert self.depth in [121]
        if self.depth == 121:
            num_init_features = 64
            growth_rate = 32
            block_config = (6, 12, 24)
            self.in_dim = [64, 256, 512, 1024]
        bn_size = 4
        drop_rate = 0

        # First convolution
        self.conv0 = A3DConv(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False, enable_shift=False)
        self.norm0 = build_norm_layer(norm_cfg, num_init_features, postfix=1)[1]
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate,
                                dimension=self.n_cts, memory_efficient=memory_efficient)
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
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

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

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        x = self.denseblock1(x)
        # x = self.normrelu1(x)
        redc1 = self.reductionz1(x)
        x = self.transition1(x)


        x = self.denseblock2(x)
        # x = self.normrelu2(x)
        redc2 = self.reductionz2(x)
        x = self.transition2(x)


        x = self.denseblock3(x)
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
        # print(key)
        
    def freeze(self):
        for name, param in self.named_parameters():
            print('freezing', name)
            param.requires_grad = False




@BACKBONES.register_module
class Trans_with_A3D(nn.Module):
    def __init__(self,
                 out_dim=256,
                 n_cts=3,
                 fpn_finest_layer=1,
                 memory_efficient=True):
        super().__init__()
        self.depth = 121
        self.feature_upsample = True
        self.fpn_finest_layer = fpn_finest_layer
        self.out_dim = out_dim
        self.n_cts = n_cts
        self.mid_ct = n_cts // 2

        assert self.depth in [121]
        if self.depth == 121:
            num_init_features = 64
            growth_rate = 32
            block_config = (6, 12, 24)
            self.in_dim = [64, 256, 512, 1024]
        bn_size = 4
        drop_rate = 0

        # First convolution
        self.conv0 = A3DConv(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False, enable_shift=False)
        self.norm0 = build_norm_layer(norm_cfg, num_init_features, postfix=1)[1]
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate,
                                dimension=self.n_cts, memory_efficient=memory_efficient)
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
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

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




        # Trans
        # self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
        self.squeeze_block1 = FCUDown(inplanes=256, outplanes=384, dw_stride=8)
        self.squeeze_block2 = FCUDown(inplanes=512, outplanes=384, dw_stride=4)
        self.squeeze_block3 = FCUDown(inplanes=1024, outplanes=384, dw_stride=2)
        # self.trans_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
        self.expand_block1 = FCUUp(inplanes=384, outplanes=256, up_stride=8)
        self.expand_block2 = FCUUp(inplanes=384, outplanes=512, up_stride=4)
        self.expand_block3 = FCUUp(inplanes=384, outplanes=1024, up_stride=2)
        self.trans_block1 = Block(dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                  drop=0.0, attn_drop=0.0, drop_path=0.001)
        self.trans_block2 = Block(dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                  drop=0.0, attn_drop=0.0, drop_path=0.001)
        self.trans_block3 = Block(dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                  drop=0.0, attn_drop=0.0, drop_path=0.001)
        if Sep_pre_Conv:
            self.trans_patch_conv1 = Sep_2D_Conv(256, 384, kernel_size=(8, 8), stride=(8, 8), padding=0, num_slices=7)
            self.trans_patch_conv2 = Sep_2D_Conv(512, 384, kernel_size=(4, 4), stride=(4, 4), padding=0, num_slices=7)
            self.trans_patch_conv3 = Sep_2D_Conv(1024, 384, kernel_size=(2, 2), stride=(2, 2), padding=0, num_slices=7)
        else:
            self.trans_patch_conv1 = nn.Conv3d(256, 384, kernel_size=(7, 8, 8), stride=(1, 8, 8), padding=0)
            self.trans_patch_conv2 = nn.Conv3d(512, 384, kernel_size=(7, 4, 4), stride=(1, 4, 4), padding=0)
            self.trans_patch_conv3 = nn.Conv3d(1024, 384, kernel_size=(7, 2, 2), stride=(1, 2, 2), padding=0)

        self.fusion_block1 = simple_ConvBlock(inplanes=256, outplanes=256)
        self.fusion_block2 = simple_ConvBlock(inplanes=512, outplanes=512)
        self.fusion_block3 = simple_ConvBlock(inplanes=1024, outplanes=1024)
        Unet_FPN = Unet_FPN_default
        self.Unet_FPN = Unet_FPN
        # unet decoder
        if Unet_FPN:
            self.lateral4 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
            self.up2 = up(in_ch=256 * 2, out_ch=512)
            self.up3 = up(in_ch=512 * 2, out_ch=256)




    def forward(self, x,thickness=None):
        x=x.contiguous()
        x = self.conv0(x)  # 4 1 7 512 512 --> 4 64 7 256 256,
        x = self.norm0(x)  # 4 64 7 256 256
        x = self.relu0(x)  # 4 64 7 256 256
        x = self.pool0(x) # 4 64 7 256 256


        ###################FPN1###############
        x = self.denseblock1(x)  # 4 64 7 256 256->4 256 7 128 128,
        B, C, S, H, W = x.shape
        # apply 2d denseblock to each slice respectivly and then concat them.
        # x = self.normrelu1(x)

        # trans block1 start
        x_t = self.trans_patch_conv1(x).contiguous()
        x_t = x_t.flatten(2).transpose(1, 2).contiguous()
        #
        cls_token = nn.Parameter(torch.zeros(B, 1, 384)).cuda().contiguous()
        x_t = torch.cat([cls_token, x_t], dim=1).contiguous()
        # x_t=x
        x_st = self.squeeze_block1(x, x_t).contiguous()
        x_t = self.trans_block1(x_st).contiguous().contiguous()
        x_t_r = self.expand_block1(x_t, H // 8, W // 8).contiguous()
        x_t_r = x_t_r.contiguous()
        # trans block1 end

        redc1 = self.reductionz1(x).squeeze(2).contiguous()  # 3D conv 7,1,1:   4 256 7 128 128->4 256 1 128 128
        redc1 = self.fusion_block1(redc1, x_t_r).contiguous()

        ###############FPN2#####################
        x = self.transition1(x).contiguous()  # ->3D conv 4 128 7 64 64,

        x = self.denseblock2(x).contiguous()  # ->4 512 7 64 64
        # x = self.normrelu2(x)

        B, C, S, H, W = x.shape
        # apply 2d denseblock to each slice respectivly and then concat them.
        # x = self.normrelu1(x)

        # trans block2 start
        cls_token = x_t[:, 0, :].unsqueeze(1).contiguous()
        x_t = self.trans_patch_conv2(x).flatten(2).transpose(1, 2).contiguous()
        x_t = torch.cat([cls_token, x_t], dim=1).contiguous()
        x_st = self.squeeze_block2(x, x_t).contiguous()
        x_t = self.trans_block2(x_st).contiguous()
        x_t_r = self.expand_block2(x_t, H // 4, W // 4).contiguous()
        x_t_r = x_t_r.contiguous()
        # trans block2 end

        redc2 = self.reductionz2(x).squeeze(2).contiguous()  # 4 512 1 64 64
        redc2 = self.fusion_block2(redc2, x_t_r).contiguous()

        ###############FPN3#####################
        x = self.transition2(x).contiguous()  # 4 256 7 32 32

        x = self.denseblock3(x).contiguous()  # 4 1024 7 32 32
        B, C, S, H, W = x.shape
        # apply 2d denseblock to each slice respectivly and then concat them.
        # x = self.normrelu1(x)

        # trans block3 start
        cls_token = x_t[:, 0, :].unsqueeze(1).contiguous()
        x_t = self.trans_patch_conv3(x).flatten(2).transpose(1, 2).contiguous()
        x_t = torch.cat([cls_token, x_t], dim=1).contiguous()
        x_st = self.squeeze_block3(x, x_t).contiguous()
        x_t = self.trans_block3(x_st).contiguous()
        x_t_r = self.expand_block3(x_t, H // 2, W // 2).contiguous()
        x_t_r=x_t_r.contiguous()
        # trans block3 end

        # x = self.normrelu3(x)

        redc3 = self.reductionz3(x).squeeze(2).contiguous()  # 4 1024 1 32 32
        redc3 = self.fusion_block3(redc3, x_t_r).contiguous()
        # truncated since here since we find it works better in DeepLesion
        # ts3 = self.transition3(db3)
        # db4 = self.denseblock4(ts3)

        # if self.feature_upsample:
        ftmaps = [None, redc1.squeeze(2), redc2.squeeze(2), redc3.squeeze(2)]
        x = self.lateral4(ftmaps[-1])  # 1024c->512c
        if self.Unet_FPN:
            # # Unet decoder
            for p in range(3, self.fpn_finest_layer - 1, -1):
                # x = F.interpolate(x, scale_factor=2, mode="nearest")
                y = ftmaps[p - 1]
                lateral = getattr(self, 'up%d' % p)(x, y)
                # x += lateral
                x = lateral
            return [x]


        # ori FPN
        else:
            for p in range(3, self.fpn_finest_layer - 1, -1):
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                y = ftmaps[p - 1]
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
                state_dict1[new_key] = state_dict[key]  # .unsqueeze(2)#.repeat((1,1,self.n_cts,1,1))/self.n_cts
                if t0 == 3:
                    state_dict1[new_key] = state_dict1[new_key][:, 1:2, ...]
            else:
                state_dict1[new_key] = state_dict[key]

        key = self.load_state_dict(state_dict1, strict=False)
        # print(key)

    def freeze(self):
        for name, param in self.named_parameters():
            print('freezing', name)
            param.requires_grad = False


@BACKBONES.register_module
class Patch_Trans_with_A3D(nn.Module):
    def __init__(self,
                 out_dim=256,
                 n_cts=3,
                 fpn_finest_layer=1,
                 memory_efficient=True):
        super().__init__()
        self.depth = 121
        self.feature_upsample = True
        self.fpn_finest_layer = fpn_finest_layer
        self.out_dim = out_dim
        self.n_cts = n_cts
        self.mid_ct = n_cts // 2

        assert self.depth in [121]
        if self.depth == 121:
            num_init_features = 64
            growth_rate = 32
            block_config = (6, 12, 24)
            self.in_dim = [64, 256, 512, 1024]
        bn_size = 4
        drop_rate = 0

        # First convolution
        self.conv0 = A3DConv(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False, enable_shift=False)
        self.norm0 = build_norm_layer(norm_cfg, num_init_features, postfix=1)[1]
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate,
                                dimension=self.n_cts, memory_efficient=memory_efficient)
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
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

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




        # Trans
        # self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
        self.squeeze_block1 = FCUDown(inplanes=256, outplanes=384, dw_stride=8)
        self.squeeze_block2 = FCUDown(inplanes=512, outplanes=384, dw_stride=4)
        self.squeeze_block3 = FCUDown(inplanes=1024, outplanes=384, dw_stride=2)
        # self.trans_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
        self.expand_block1 = FCUUp(inplanes=384, outplanes=256, up_stride=8)
        self.expand_block2 = FCUUp(inplanes=384, outplanes=512, up_stride=4)
        self.expand_block3 = FCUUp(inplanes=384, outplanes=1024, up_stride=2)
        self.trans_block1 = Block(dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                  drop=0.0, attn_drop=0.0, drop_path=0.001)
        self.trans_block2 = Block(dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                  drop=0.0, attn_drop=0.0, drop_path=0.001)
        self.trans_block3 = Block(dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                  drop=0.0, attn_drop=0.0, drop_path=0.001)
        if Sep_pre_Conv:
            self.trans_patch_conv1 = Sep_2D_Conv(256, 384, kernel_size=(8, 8), stride=(8, 8), padding=0, num_slices=7)
            self.trans_patch_conv2 = Sep_2D_Conv(512, 384, kernel_size=(4, 4), stride=(4, 4), padding=0, num_slices=7)
            self.trans_patch_conv3 = Sep_2D_Conv(1024, 384, kernel_size=(2, 2), stride=(2, 2), padding=0, num_slices=7)
        else:
            self.trans_patch_conv1 = nn.Conv3d(256, 384, kernel_size=(7, 8, 8), stride=(1, 8, 8), padding=0)
            self.trans_patch_conv2 = nn.Conv3d(512, 384, kernel_size=(7, 4, 4), stride=(1, 4, 4), padding=0)
            self.trans_patch_conv3 = nn.Conv3d(1024, 384, kernel_size=(7, 2, 2), stride=(1, 2, 2), padding=0)

        self.fusion_block1 = simple_ConvBlock(inplanes=256, outplanes=256)
        self.fusion_block2 = simple_ConvBlock(inplanes=512, outplanes=512)
        self.fusion_block3 = simple_ConvBlock(inplanes=1024, outplanes=1024)
        Unet_FPN = Unet_FPN_default
        self.Unet_FPN = Unet_FPN
        # unet decoder
        if Unet_FPN:
            self.lateral4 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
            self.up2 = up(in_ch=256 * 2, out_ch=512)
            self.up3 = up(in_ch=512 * 2, out_ch=256)




    def forward(self, x,thickness=None):
        x=x.contiguous()
        x = self.conv0(x)  # 4 1 7 512 512 --> 4 64 7 256 256,
        x = self.norm0(x)  # 4 64 7 256 256
        x = self.relu0(x)  # 4 64 7 256 256
        x = self.pool0(x) # 4 64 7 256 256


        ###################FPN1###############
        x = self.denseblock1(x)  # 4 64 7 256 256->4 256 7 128 128,
        B, C, S, H, W = x.shape
        # apply 2d denseblock to each slice respectivly and then concat them.
        # x = self.normrelu1(x)

        # trans block1 start
        x_t = self.trans_patch_conv1(x).contiguous()
        x_t = x_t.flatten(2).transpose(1, 2).contiguous()
        #
        cls_token = nn.Parameter(torch.zeros(B, 1, 384)).cuda().contiguous()
        x_t = torch.cat([cls_token, x_t], dim=1).contiguous()
        # x_t=x
        x_st = self.squeeze_block1(x, x_t).contiguous()
        x_t = self.trans_block1(x_st).contiguous().contiguous()
        x_t_r = self.expand_block1(x_t, H // 8, W // 8).contiguous()
        x_t_r = x_t_r.contiguous()
        # trans block1 end

        redc1 = self.reductionz1(x).squeeze(2).contiguous()  # 3D conv 7,1,1:   4 256 7 128 128->4 256 1 128 128
        redc1 = self.fusion_block1(redc1, x_t_r).contiguous()

        ###############FPN2#####################
        x = self.transition1(x).contiguous()  # ->3D conv 4 128 7 64 64,

        x = self.denseblock2(x).contiguous()  # ->4 512 7 64 64
        # x = self.normrelu2(x)

        B, C, S, H, W = x.shape
        # apply 2d denseblock to each slice respectivly and then concat them.
        # x = self.normrelu1(x)

        # trans block2 start
        cls_token = x_t[:, 0, :].unsqueeze(1).contiguous()
        x_t = self.trans_patch_conv2(x).flatten(2).transpose(1, 2).contiguous()
        x_t = torch.cat([cls_token, x_t], dim=1).contiguous()
        x_st = self.squeeze_block2(x, x_t).contiguous()
        x_t = self.trans_block2(x_st).contiguous()
        x_t_r = self.expand_block2(x_t, H // 4, W // 4).contiguous()
        x_t_r = x_t_r.contiguous()
        # trans block2 end

        redc2 = self.reductionz2(x).squeeze(2).contiguous()  # 4 512 1 64 64
        redc2 = self.fusion_block2(redc2, x_t_r).contiguous()

        ###############FPN3#####################
        x = self.transition2(x).contiguous()  # 4 256 7 32 32

        x = self.denseblock3(x).contiguous()  # 4 1024 7 32 32
        B, C, S, H, W = x.shape
        # apply 2d denseblock to each slice respectivly and then concat them.
        # x = self.normrelu1(x)

        # trans block3 start
        cls_token = x_t[:, 0, :].unsqueeze(1).contiguous()
        x_t = self.trans_patch_conv3(x).flatten(2).transpose(1, 2).contiguous()
        x_t = torch.cat([cls_token, x_t], dim=1).contiguous()
        x_st = self.squeeze_block3(x, x_t).contiguous()
        x_t = self.trans_block3(x_st).contiguous()
        x_t_r = self.expand_block3(x_t, H // 2, W // 2).contiguous()
        x_t_r=x_t_r.contiguous()
        # trans block3 end

        # x = self.normrelu3(x)

        redc3 = self.reductionz3(x).squeeze(2).contiguous()  # 4 1024 1 32 32
        redc3 = self.fusion_block3(redc3, x_t_r).contiguous()
        # truncated since here since we find it works better in DeepLesion
        # ts3 = self.transition3(db3)
        # db4 = self.denseblock4(ts3)

        # if self.feature_upsample:
        ftmaps = [None, redc1.squeeze(2), redc2.squeeze(2), redc3.squeeze(2)]
        x = self.lateral4(ftmaps[-1])  # 1024c->512c
        if self.Unet_FPN:
            # # Unet decoder
            for p in range(3, self.fpn_finest_layer - 1, -1):
                # x = F.interpolate(x, scale_factor=2, mode="nearest")
                y = ftmaps[p - 1]
                lateral = getattr(self, 'up%d' % p)(x, y)
                # x += lateral
                x = lateral
            return [x]


        # ori FPN
        else:
            for p in range(3, self.fpn_finest_layer - 1, -1):
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                y = ftmaps[p - 1]
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
                state_dict1[new_key] = state_dict[key]  # .unsqueeze(2)#.repeat((1,1,self.n_cts,1,1))/self.n_cts
                if t0 == 3:
                    state_dict1[new_key] = state_dict1[new_key][:, 1:2, ...]
            else:
                state_dict1[new_key] = state_dict[key]

        key = self.load_state_dict(state_dict1, strict=False)
        # print(key)

    def freeze(self):
        for name, param in self.named_parameters():
            print('freezing', name)
            param.requires_grad = False