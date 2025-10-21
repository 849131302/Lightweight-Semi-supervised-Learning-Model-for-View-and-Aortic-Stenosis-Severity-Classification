import math
import os
import random

import numpy as np
import timm
import torch.nn.functional as F
import torch
import copy
import torch.nn as nn
from typing import List, Tuple

from fvcore.nn import FlopCountAnalysis, flop_count_table
from pytorch_wavelets import DWTForward
from timm import optim
from torch.utils.data import DistributedSampler, SequentialSampler, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from torchvision.transforms import InterpolationMode

from augmennt import TargetedStrongAugment
from models import fastvit_t8
from models.fastvit import fastvit_t12


class RepBlock(nn.Module):
    """
    MobileOne-style residual blocks, including residual joins and re-parameterization convolutions
    这个类主要实现了 MobileOne 风格的残差块，包括残差连接和可重参数化卷积。在推理模式下，直接使用 Conv2d 进行前向传播，
    而在训练模式下，通过 Re-parameterization 方法将多个卷积分支合并为一个可重参数化的卷积层。在 Re-parameterization 过程中，将不需要的分支删除，以减少模型的计算量。
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            groups: int = 1,
            inference_mode: bool = False,
            rbr_conv_kernel_list: List[int] = [7, 3],
            use_bn_conv: bool = False,
            act_layer: nn.Module = nn.GELU,
            #skip_include_bn: bool = True,
            skip_include_bn: bool = False,
    ) -> None:
        """Construct a Re-parameterization module.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for convolution.
        :param groups: Number of groups for convolution.
        :param inference_mode: Whether to use inference mode.
        :param rbr_conv_kernel_list: List of kernel sizes for re-parameterizable convolutions.
        :param use_bn_conv: Whether the bn is in front of conv, if false, conv is in front of bn
        :param act_layer: Activation layer.
        :param skip_include_bn: Whether to include bn in skip connection.
        """
        super(RepBlock, self).__init__()

        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rbr_conv_kernel_list = sorted(rbr_conv_kernel_list, reverse=True)
        self.num_conv_branches = len(self.rbr_conv_kernel_list)
        self.kernel_size = self.rbr_conv_kernel_list[0]
        self.use_bn_conv = use_bn_conv
        self.skip_include_bn = skip_include_bn

        self.activation = act_layer()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.kernel_size // 2,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            if out_channels == in_channels and stride == 1:
                if self.skip_include_bn:
                    # Use residual connections that include BN
                    self.rbr_skip = nn.BatchNorm2d(num_features=in_channels)
                else:
                    # Use residual connections
                    self.rbr_skip = nn.Identity()
            else:
                # Use residual connections
                self.rbr_skip = None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for kernel_size in self.rbr_conv_kernel_list:
                if self.use_bn_conv:
                    rbr_conv.append(
                        self._bn_conv(
                            in_chans=in_channels,
                            out_chans=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            groups=groups,
                        )
                    )
                else:
                    rbr_conv.append(
                        self._conv_bn(
                            in_chans=in_channels,
                            out_chans=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            groups=groups,
                        )
                    )

            self.rbr_conv = nn.ModuleList(rbr_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Other branches
        out = identity_out
        for ix in range(self.num_conv_branches):
            out = out + self.rbr_conv[ix](x)
        return self.activation(out)

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0].conv.in_channels,
            out_channels=self.rbr_conv[0].conv.out_channels,
            kernel_size=self.rbr_conv[0].conv.kernel_size,
            stride=self.rbr_conv[0].conv.stride,
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation,
            groups=self.rbr_conv[0].conv.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_skip_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            if self.use_bn_conv:
                _kernel, _bias = self._fuse_bn_conv_tensor(self.rbr_conv[ix])
            else:
                _kernel, _bias = self._fuse_conv_bn_tensor(self.rbr_conv[ix])
            # pad kernel
            if _kernel.shape[-1] < self.kernel_size:
                pad = (self.kernel_size - _kernel.shape[-1]) // 2
                _kernel = torch.nn.functional.pad(_kernel, [pad, pad, pad, pad])

            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_identity
        bias_final = bias_conv + bias_identity
        return kernel_final, bias_final

    def _fuse_skip_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param branch: skip branch, maybe include bn layer
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """

        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros(
                (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                dtype=self.rbr_conv[0].conv.weight.dtype,
                device=self.rbr_conv[0].conv.weight.device,
            )
            for i in range(self.in_channels):
                kernel_value[
                    i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                ] = 1
            self.id_tensor = kernel_value
        if isinstance(branch, nn.Identity):
            kernel = self.id_tensor
            return kernel, torch.zeros(
                (self.in_channels),
                dtype=self.rbr_conv[0].conv.weight.dtype,
                device=self.rbr_conv[0].conv.weight.device,
            )
        else:
            assert isinstance(
                branch, nn.BatchNorm2d
            ), "Make sure the module in skip is nn. BatchNorm2d"
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_conv_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """先bn,后conv

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = gamma / std
        t = torch.stack([t] * (kernel.shape[0] * kernel.shape[1] // t.shape[0]), dim=0).reshape(-1,self.in_channels // self.groups,1, 1)
        t_beta = torch.stack([beta] * (kernel.shape[0] * kernel.shape[1] // beta.shape[0]), dim=0).reshape(-1,self.in_channels // self.groups,1, 1)
        t_running_mean = torch.stack([running_mean] * (kernel.shape[0] * kernel.shape[1] // running_mean.shape[0]),dim=0).reshape(-1, self.in_channels // self.groups, 1, 1)
        return kernel * t, torch.sum(kernel* (t_beta - t_running_mean * t),dim=(1, 2, 3),)

    def _fuse_conv_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """First conv, then bn

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """

        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            stride: int,
            padding: int,
            groups: int,
    ) -> nn.Sequential:
        """First conv, then bn

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=out_chans))
        return mod_list

    def _bn_conv(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int,
            stride: int,
            padding: int,
            groups: int,
    ) -> nn.Sequential:
        """Add bn first, then conv"""
        mod_list = nn.Sequential()
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=in_chans))
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )
        return mod_list




'''
class ERep_Down_wt(nn.Module):
    def __init__(self, in_channels, out_channels,act_layer:nn.Module,inference_mode: bool,J=1):
        super(ERep_Down_wt, self).__init__()
        self.wt = DWTForward(J=J, mode='zero', wave='haar')
        self.conv_bn_gelu= nn.Sequential(
        # nn.Conv2d(in_channels*4, out_channels, kernel_size=1, stride=1),
        # nn.BatchNorm2d(out_channels),
        # nn.GELU(),
        # RepBlock(
        # in_channels=in_channels*4,
        # out_channels=in_channels*4,
        # rbr_conv_kernel_list=[3,1],
        # stride=1,
        # groups=in_channels*4,
        # inference_mode=inference_mode,
        # act_layer=act_layer,
        # ),
        RepBlock(
        in_channels=in_channels*4,
        out_channels=out_channels,
        rbr_conv_kernel_list=[1],
        stride=1,
        groups= 1,
        inference_mode=inference_mode,
        act_layer=act_layer,
        use_bn_conv=False,
        skip_include_bn=False),
        )
    def forward(self, x):
        #x = self.conv_bn_gelu1(x)
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_gelu(x)
        return x'''




class ERep_Down_wt(nn.Module):
    def __init__(self, in_channels, out_channels,act_layer:nn.Module,inference_mode: bool,J=1):
        super(ERep_Down_wt, self).__init__()
        self.conv_bn_gelu1 = nn.Sequential(
            RepBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                rbr_conv_kernel_list=[3,1],
                stride=1,
                groups=in_channels,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
        )
        self.wt = DWTForward(J=J, mode='zero', wave='haar')
        self.conv_bn_gelu2 = nn.Sequential(
            # nn.Conv2d(in_channels*4, out_channels, kernel_size=1, stride=1),
            # nn.BatchNorm2d(out_channels),
            # nn.GELU(),
            RepBlock(
                in_channels=in_channels * 4,
                out_channels=out_channels,
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
                use_bn_conv=False,
                skip_include_bn=False),
        )

    def forward(self, x):
        x = self.conv_bn_gelu1(x)
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_gelu2(x)
        return x


# class Rep_Down_wt(nn.Module):
#     def __init__(self, in_channels, out_channels,act_layer:nn.Module,inference_mode: bool,J=1):
#         super(Rep_Down_wt, self).__init__()
#         self.wt = DWTForward(J=J, mode='zero', wave='haar')
#         self.conv_bn_gelu = nn.Sequential(
#         # nn.Conv2d(in_channels*4, out_channels, kernel_size=1, stride=1),
#         # nn.BatchNorm2d(out_channels),
#         # nn.GELU(),
#         RepBlock(
#         in_channels=in_channels*4,
#         out_channels=in_channels*4,
#         rbr_conv_kernel_list=[7, 5],
#         stride=1,
#         groups=in_channels*4,
#         inference_mode=inference_mode,
#         act_layer=act_layer,
#         ),
#         RepBlock(
#         in_channels=in_channels*4,
#         out_channels=out_channels,
#         rbr_conv_kernel_list=[1],
#         stride=1,
#         groups= 1,
#         inference_mode=inference_mode,
#         act_layer=act_layer,
#         use_bn_conv=False,
#         skip_include_bn=False),
#         )
#     def forward(self, x):
#         yL, yH = self.wt(x)
#         y_HL = yH[0][:,:,0,::]
#         y_LH = yH[0][:,:,1,::]
#         y_HH = yH[0][:,:,2,::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv_bn_gelu(x)
#         return x
class Rep_Down_wt(nn.Module):
    def __init__(self, in_channels, out_channels,act_layer:nn.Module,inference_mode: bool,J=1):
        super(Rep_Down_wt, self).__init__()
        self.conv_bn_gelu1 = nn.Sequential(
            RepBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                rbr_conv_kernel_list=[7,5],
                stride=1,
                groups=in_channels,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
        )
        self.wt = DWTForward(J=J, mode='zero', wave='haar')
        self.conv_bn_gelu2 = nn.Sequential(
        # nn.Conv2d(in_channels*4, out_channels, kernel_size=1, stride=1),
        # nn.BatchNorm2d(out_channels),
        # nn.GELU(),
        RepBlock(
        in_channels=in_channels*4,
        out_channels=out_channels,
        rbr_conv_kernel_list=[1],
        stride=1,
        groups=1,
        inference_mode=inference_mode,
        act_layer=act_layer,
        use_bn_conv=False,
        skip_include_bn=False),
        )
    def forward(self, x):
        x = self.conv_bn_gelu1(x)
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_gelu2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class AttnTokenMixer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        num_heads: int,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_chans)
        self.attn = Attention(
            in_chans,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
        )
        #self.attn = MLCA(dim=num_heads,)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.attn(self.norm(x))
        x = x.permute(0, 2, 1).reshape(B, C, Hp, Wp).contiguous()
        return x



class LSKA(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        if kernel_size== 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif kernel_size== 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif kernel_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif kernel_size== 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
        elif kernel_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
        elif kernel_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn
class LSKAttention(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKA(dim, kernel_size)
        self.proj_2 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class RepLSKABlock(nn.Module):
    def __init__(self, dim, kernel_size,act_layer:nn.Module,inference_mode):
        super().__init__()
        # self.proj_1 = nn.Conv2d(dim, dim, 1)
        # self.activation = nn.GELU()
        self.Attention = nn.Sequential(
            RepBlock(
                    in_channels=dim,
                    out_channels=dim,
                    rbr_conv_kernel_list=[1],
                    stride=1,
                    groups=1,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_bn_conv=False,
                    skip_include_bn=False
                ),
        LSKA(dim, kernel_size),
        RepBlock(
                    in_channels=dim,
                    out_channels=dim,
                    rbr_conv_kernel_list=[1],
                    stride=1,
                    groups=1,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_bn_conv=False,
                    skip_include_bn=False
                ),)
    def forward(self, x):
        shorcut = x.clone()
        x = self.Attention(x)
        x = x + shorcut
        return x



class RepLKA(nn.Module):
    def __init__(self, dim,act_layer:nn.Module,inference_mode:bool):
        super().__init__()
        self.Attention = nn.Sequential(
            RepBlock(
                in_channels=dim,
                out_channels=dim,
                rbr_conv_kernel_list=[7,5],
                stride=1,
                groups=dim,
                inference_mode=inference_mode,
                act_layer=act_layer,
                use_bn_conv=False,
                skip_include_bn=False
            ),
            # RepBlock(
            #     in_channels=dim,
            #     out_channels=dim,
            #     rbr_conv_kernel_list=[3, 1],
            #     stride=1,
            #     groups=dim,
            #     inference_mode=inference_mode,
            #     act_layer=act_layer,
            #     use_bn_conv=False,
            #     skip_include_bn=False
            # ),
        RepBlock(
            in_channels=dim,
            out_channels=dim,
            rbr_conv_kernel_list=[1],
            stride=1,
            groups=1,
            inference_mode=inference_mode,
            act_layer=act_layer,
            use_bn_conv=False,
            skip_include_bn=False
        ),
        )
    def forward(self,x):
        u=x.clone()
        attn = self.Attention(x)
        return u * attn
class RepLKABlock(nn.Module):
    def __init__(self, dim,act_layer:nn.Module,inference_mode):
        super().__init__()
        # self.proj_1 = nn.Conv2d(dim, dim, 1)
        # self.activation = nn.GELU()
        self.AttentionBlock = nn.Sequential(RepBlock(
                    in_channels=dim,
                    out_channels=dim,
                    rbr_conv_kernel_list=[1],
                    stride=1,
                    groups=1,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_bn_conv=False,
                    skip_include_bn=False
                ),
        RepLKA(dim=dim,act_layer=act_layer, inference_mode=inference_mode),
        RepBlock(
                    in_channels=dim,
                    out_channels=dim,
                    rbr_conv_kernel_list=[1],
                    stride=1,
                    groups=1,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_bn_conv=False,
                    skip_include_bn=False
                ),)
    def forward(self, x):
        shorcut = x.clone()
        x = self.AttentionBlock(x)
        x = x + shorcut
        return x




class CSE(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CSE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channel,channel // ratio, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channel// ratio,channel, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.squeeze(x)
        z = self.excitation(y).view(-1, c, 1, 1)
        return x * z.expand_as(x)



class RepCSE(nn.Module):
    def __init__(self, channel, ratio=16,inference_mode=False):
        super(RepCSE, self).__init__()
        self.excitation = nn.Sequential(
            RepBlock(
                in_channels=channel,
                out_channels=channel // ratio,
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=nn.GELU,
                use_bn_conv=False,
                skip_include_bn=False
            ),
            RepBlock(
                in_channels=channel // ratio,
                out_channels=channel,
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=nn.Sigmoid,
                use_bn_conv=False,
                skip_include_bn=False
            ),
        )
    def forward(self, x):
        z = self.excitation(x)
        return x * z.expand_as(x)



class SE(nn.Module):
    def __init__(self, channel, reduction=24):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SLWBlock(nn.Module):
    def __init__(
        self,
        chans: int,
        num_heads: int,
        inference_mode: bool,
        act_layer: nn.Module,
        use_attn: bool,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        if use_attn:
            #AttnTokenMixer
            #self.token_mixer = AttnTokenMixer(chans, num_heads)

            # self.token_mixer = nn.Sequential(
            #     RepBlock(
            #         in_channels=chans,
            #         out_channels=chans,
            #         rbr_conv_kernel_list=[3],
            #         stride=1,
            #         groups=chans,
            #         inference_mode=inference_mode,
            #         act_layer=nn.Identity,
            #         use_bn_conv=False,
            #         skip_include_bn=False
            #     ),
            #     LSKAttention(dim=chans, kernel_size=35),
            # )
            self.token_mixer = nn.Sequential(
                # RepBlock(
                #     in_channels=chans,
                #     out_channels=chans,
                #     rbr_conv_kernel_list=[3],
                #     stride=1,
                #     groups=chans,
                #     inference_mode=inference_mode,
                #     act_layer=act_layer,
                #     use_bn_conv=False,
                #     skip_include_bn=False
                # ),
                #RepLSKABlock(dim=chans, kernel_size=23,act_layer=act_layer,inference_mode=inference_mode)
                RepLKA(dim=chans,act_layer=act_layer, inference_mode=inference_mode)
            )

        else:
            # RepMixer
            '''深度可分离卷积：https://zhuanlan.zhihu.com/p/80041030 https://zhuanlan.zhihu.com/p/490685194
    先是depthwiseConv，本质上就是分组卷积，在深度可分离卷积中，分组卷积的组数=输入通道数=输出通道数，该部分通道数不变
    再是pointwiseConv，就是点卷积，该部分负责扩展通道数，所以其kernel_size=1，不用padding'''
            self.token_mixer =nn.Sequential(
                RepBlock(
                    in_channels=chans,
                    out_channels=chans,
                    rbr_conv_kernel_list=[3,1],
                    stride=1,
                    groups=chans,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_bn_conv=False,
                    skip_include_bn=False
                ),
                # RepBlock(
                #     in_channels=chans,
                #     out_channels=chans,
                #     rbr_conv_kernel_list=[1],
                #     stride=1,
                #     groups=1,
                #     inference_mode=inference_mode,
                #     act_layer=act_layer,
                #     use_bn_conv=False,
                #     skip_include_bn=False
                # ),
            )
        mid_chans = chans * expand_ratio
        #conv_ffn
        # self.conv_ffn = nn.Sequential(
        #     nn.Conv2d(chans, chans, kernel_size=7, padding=3, groups=chans, bias=False),
        #     nn.BatchNorm2d(chans),
        #     nn.Conv2d(chans, mid_chans, kernel_size=1, padding=0),
        #     act_layer(),
        #     nn.Conv2d(mid_chans, chans, kernel_size=1, padding=0),
        # )
        self.conv_ffn = nn.Sequential(
            RepBlock(
                in_channels=chans,
                out_channels=chans,
                rbr_conv_kernel_list=[7,5],
                #rbr_conv_kernel_list=[3,1],
                stride=1,
                groups=chans,
                inference_mode=inference_mode,
                act_layer=act_layer,
                use_bn_conv=False,
                skip_include_bn=False
            ),
            RepBlock(
                in_channels=chans,
                out_channels=mid_chans,
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,#nn.Identity
                use_bn_conv=False,
                skip_include_bn=False
            ),
            RepBlock(
                in_channels=mid_chans,
                out_channels=chans,
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
                use_bn_conv=False,
                skip_include_bn=False
            ),

        )

    def forward(self, x):
        x = self.token_mixer(x)
        x = x + self.conv_ffn(x)
        return x


class LWStage(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        use_attn: bool,
        num_heads: int,
        num_blocks_per_stage: int,
        inference_mode: bool,
        act_layer: nn.Module,
        expand_ratio: int,
        use_erep_hwd: bool,
    ) -> None:
        """
        Constructs a FastStage

        :param in_chans: Number of input channels.
        :param out_chans: Number of output channels.
        :param num_heads: Number of heads for attention If use_attn is True.
        :param use_attn: Whether to use attention.
        :param num_blocks_per_stage: Number of blocks per stage.
        :param inference_mode: Whether to use inference mode.
        :param act_layer: Activation layer.
        :param expand_ratio: Expansion ratio in conv_ffn.
        :param use_patch_embed: Whether to use patch embedding.
        """
        super().__init__()
        self.num_blocks_per_stage = num_blocks_per_stage
        # self.RepDWT = nn.Sequential(
        #     # RepBlock(
        #     #     in_channels=in_chans,
        #     #     out_channels=in_chans,
        #     #     rbr_conv_kernel_list=[7,5],
        #     #     stride=1,
        #     #     groups=in_chans,
        #     #     inference_mode=inference_mode,
        #     #     act_layer=act_layer,
        #     # ),
        #     Rep_Down_wt(in_channels=in_chans, out_channels=out_chans,act_layer=act_layer,inference_mode=inference_mode,J=1),
        # )

        '''if use_rep_hwd:
            self.RepDWT = nn.Sequential(Rep_Down_wt(in_channels=in_chans, out_channels=out_chans, act_layer=act_layer,inference_mode=inference_mode, J=1),)
        else:
            self.RepDWT = nn.Identity()'''
        if use_erep_hwd:
            self.RepDWT = nn.Sequential(ERep_Down_wt(in_channels=in_chans, out_channels=out_chans, act_layer=act_layer,inference_mode=inference_mode, J=1),)
            '''self.RepDWT =nn.Sequential(
            RepBlock(
                in_channels=in_chans,
                out_channels=in_chans,
                rbr_conv_kernel_list=[3,1],
                stride=1,
                groups=in_chans,
                inference_mode=inference_mode,
                act_layer=act_layer,
                use_bn_conv=False,
                skip_include_bn=False
            ),
            RepBlock(
                in_channels=in_chans,
                out_channels=out_chans,
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,#nn.Identity
                use_bn_conv=False,
                skip_include_bn=False
            ),
            )'''
        else:
            self.RepDWT = nn.Sequential(Rep_Down_wt(in_channels=in_chans, out_channels=out_chans, act_layer=act_layer,inference_mode=inference_mode, J=1),)
            '''self.RepDWT = nn.Sequential(
                RepBlock(
                    in_channels=in_chans,
                    out_channels=in_chans,
                    rbr_conv_kernel_list=[7, 5],
                    stride=2,
                    groups=in_chans,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_bn_conv=False,
                    skip_include_bn=False
                ),
                RepBlock(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    rbr_conv_kernel_list=[1],
                    stride=1,
                    groups=1,
                    inference_mode=inference_mode,
                    act_layer=act_layer,  # nn.Identity
                    use_bn_conv=False,
                    skip_include_bn=False
                ),
            )'''

        self.blocks = nn.Sequential(
            *[
                SLWBlock(
                    chans=out_chans,
                    num_heads=num_heads,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_attn=use_attn,
                    expand_ratio=expand_ratio,
                )
                for i in range(num_blocks_per_stage)
            ]
        )

    def forward(self, x):
        x = self.RepDWT(x)
        x = self.blocks(x)
        return x




class Light_Weight_Neteork(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        inference_mode: bool = False,
        in_chans_list: Tuple[int] = (48, 48, 96, 192),
        out_chans_list: Tuple[int] = (48, 96, 192, 384),
        blocks_per_stage: Tuple[int] = (2, 2, 6, 2),
        expand_ratio: Tuple[int] = (4, 4, 4, 4),
        use_attn: Tuple[bool] = (False, False,False,False),
        act_layer: nn.Module = nn.GELU,
        #use_rep_hwd: Tuple[bool] = (False, True, True, True),
        use_erep_hwd: Tuple[bool] = (True, False, False, False),
    ) -> None:
        """
        Constructs a FastVit model

        :param num_classes: Number of classes for classification head.
        :param inference_mode: Whether to use inference mode.
        :param in_chans_list: List of input channels for each stage.
        :param out_chans_list: List of output channels for each stage.
        :param blocks_per_stage: List of number of blocks for each stage.
        :param expand_ratio: List of expansion ratios for each stage.
        :param use_attn: List of whether to use attention for each stage.
        :param act_layer: Activation layer.
        """

        super().__init__()
        '''self.stem = nn.Sequential(
        # RepBlock(
        #     in_channels=3,
        #     out_channels=in_chans_list[0],
        #     rbr_conv_kernel_list=[3,1],
        #     stride=1,
        #     groups=3,
        #     inference_mode=inference_mode,
        #     act_layer=act_layer,
        # ),
        #ERep_Down_wt(in_channels=in_chans_list[0], out_channels=in_chans_list[0], act_layer=act_layer, inference_mode=inference_mode, J=1),
        #ERep_Down_wt(in_channels=3, out_channels=(in_chans_list[0])//2, act_layer=act_layer,inference_mode=inference_mode, J=1),
        # RepBlock(
        #     in_channels=in_chans_list[0],
        #     out_channels=in_chans_list[0],
        #     rbr_conv_kernel_list=[3,1],
        #     stride=1,
        #     groups=in_chans_list[0],
        #     inference_mode=inference_mode,
        #     act_layer=act_layer,
        # ),
        # RepBlock(
        #         in_channels=in_chans_list[0],
        #         out_channels=in_chans_list[0],
        #         rbr_conv_kernel_list=[3,1],
        #         stride=1,
        #         groups=in_chans_list[0],
        #         inference_mode=inference_mode,
        #         act_layer=act_layer,
        #     ),
        ERep_Down_wt(in_channels=3, out_channels=in_chans_list[0], act_layer=act_layer, inference_mode=inference_mode, J=2),
        #ERep_Down_wt(in_channels=(in_chans_list[0])//2, out_channels=in_chans_list[0], act_layer=act_layer,inference_mode=inference_mode, J=1),
        # RepBlock(
        #     in_channels=in_chans_list[0],
        #     out_channels=in_chans_list[0],
        #     rbr_conv_kernel_list=[1],
        #     stride=1,
        #     groups=1,
        #     inference_mode=inference_mode,
        #     act_layer=act_layer,
        # ),
        )'''
        self.stages = nn.Sequential(
            *(
                LWStage(
                    in_chans=in_chans_list[i],
                    out_chans=out_chans_list[i],
                    num_blocks_per_stage=blocks_per_stage[i],
                    inference_mode=inference_mode,
                    use_attn=use_attn[i],
                    num_heads=8,
                    expand_ratio=expand_ratio[i],
                    act_layer=act_layer,
                    #use_rep_hwd=use_rep_hwd[i],
                    use_erep_hwd=use_erep_hwd[i],
                )
                for i in range(len(blocks_per_stage))
            )
        )

        '''self.last_block =nn.Sequential(
            RepBlock(
            in_channels=out_chans_list[-1],
            out_channels=out_chans_list[-1],
            stride=1,
            groups=out_chans_list[-1],
            inference_mode=inference_mode,
            rbr_conv_kernel_list=[3,1],
            act_layer=act_layer,
            ),
            # RepBlock(
            # in_channels=out_chans_list[-1],
            # out_channels=out_chans_list[-1],
            # stride=1,
            # groups=1,
            # inference_mode=inference_mode,
            # rbr_conv_kernel_list=[1],
            # act_layer=act_layer,
            # ),
            #CSE(channel=out_chans_list[-1])
        )'''
        self.head = nn.Sequential(
            # RepBlock(
            # in_channels=out_chans_list[-1],
            # out_channels=out_chans_list[-1],
            # stride=1,
            # groups=out_chans_list[-1],
            # inference_mode=inference_mode,
            # rbr_conv_kernel_list=[3,1],
            # act_layer=act_layer,
            # ),
            #RepCSE(channel=out_chans_list[-1],inference_mode=inference_mode),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=out_chans_list[-1], out_features=num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        #x = self.stem(x)
        #print(x.shape)
        x = self.stages(x)
        #x = self.last_block(x)
        #print(x.shape)
        x = self.head(x)
        return x


# def lw_vit(
#     num_classes: int = 4, inference_mode: bool = False,
# ) -> nn.Module:
#     """
#     Constructs a LWVit model
#
#     :param num_classes: Number of classes for classification head.
#     :param inference_mode: Whether to use inference mode.
#     :param variant: Variant of LWVit.
#     """
#     params = {
#         "in_chans_list": (48, 48, 96, 192),
#         "out_chans_list": (48, 96, 192, 384),
#         "blocks_per_stage": (2, 2, 4, 2),
#         "expand_ratio": (3, 3, 3, 3),
#         "use_attn": (False, False, False, False,),
#         "use_patchEmb": (False, True, True, True),
#     }
#     return LWVit(
#         num_classes=num_classes, inference_mode=inference_mode, **params
#     )
def LWN(
    num_classes: int = 3, inference_mode: bool = False,
) -> nn.Module:
    """
    Constructs a LWVit model

    :param num_classes: Number of classes for classification head.
    :param inference_mode: Whether to use inference mode.
    :param variant: Variant of LWVit.
    expand_ratio 用于控制 RepBlock 内部的通道扩张，影响每个阶段 RepBlock 的内部结构。
    out_chans_list 用于定义每个阶段的输出通道数，影响整个模型的整体结构。
    """
    params = {
        # "in_chans_list": (28, 28, 56, 112),
        # "out_chans_list": (28, 56, 112, 224),
        "in_chans_list": (3, 24, 48, 96),
        "out_chans_list": (24, 48, 96, 192),
        # "in_chans_list": (3, 36, 72, 144),
        # "out_chans_list": (36, 72, 144, 288),
        "blocks_per_stage": (2, 2, 6, 2),
        "expand_ratio": (2, 2, 2, 2),
        # "blocks_per_stage": (2, 4, 6, 2),
        # "expand_ratio": (4, 4, 4, 4),
        "use_attn": (False, False, False,False),
    }
    return Light_Weight_Neteork(
        num_classes=num_classes, inference_mode=inference_mode, **params
    )
def LWN_4VIEW(
    num_classes: int = 4, inference_mode: bool = False,
) -> nn.Module:
    """
    Constructs a LWVit model

    :param num_classes: Number of classes for classification head.
    :param inference_mode: Whether to use inference mode.
    :param variant: Variant of LWVit.
    expand_ratio 用于控制 RepBlock 内部的通道扩张，影响每个阶段 RepBlock 的内部结构。
    out_chans_list 用于定义每个阶段的输出通道数，影响整个模型的整体结构。
    """
    params = {
        # "in_chans_list": (28, 28, 56, 112),
        # "out_chans_list": (28, 56, 112, 224),
        "in_chans_list": (3, 24, 48, 96),
        "out_chans_list": (24, 48, 96, 192),
        # "in_chans_list": (3, 36, 72, 144),
        # "out_chans_list": (36, 72, 144, 288),
        "blocks_per_stage": (2, 2, 6, 2),
        "expand_ratio": (2, 2, 2, 2),
        # "blocks_per_stage": (2, 4, 6, 2),
        # "expand_ratio": (4, 4, 4, 4),
        "use_attn": (False, False, False,False),
    }
    return Light_Weight_Neteork(
        num_classes=num_classes, inference_mode=inference_mode, **params
    )




def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model



if __name__ == "__main__":
    labeled_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-AS\DEV479\\train'
    unlabeled_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\unlabeled_set'# unlabeled_set
    val_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-AS\DEV479\\val'
    test_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-AS\DEV479\\test'


    def seed_everything(seed):  # 设置随机因子  设置了固定的随机因子，再次训练的时候就可以保证图片的加载顺序不会发生变化
        os.environ['PYHTONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
    seed_everything(seed=1)
    # 定义数据变换
    class TransformTwice:
        def __init__(self, weak_transform, strong_transform):
            self.weak_transform = weak_transform
            self.strong_transform = strong_transform

        def __call__(self, x):
            ulb_w = self.weak_transform(x)
            ulb_s = self.strong_transform(x)
            return ulb_w, ulb_s


    transform_labeledtrain = transforms.Compose([
        #transforms.Resize((256, 256)),
        transforms.Resize(size=(224,224), interpolation=InterpolationMode.LANCZOS),  # 32, 32
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(15),
        # transforms.RandomCrop(size=112,
        #                       padding=int(112* 0.125),
        #                       padding_mode='reflect'),
        transforms.ToTensor(),
        # transforms.Normalize(0.05, 0.12)
    ])
    weak_transform = transforms.Compose([
        transforms.Resize(size=(224,224), interpolation=InterpolationMode.LANCZOS),  # 32, 32
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(15),
        # transforms.RandomCrop(size=112,
        #                       padding=int(112 * 0.125),
        #                       padding_mode='reflect'),
        transforms.ToTensor(),
        # transforms.Normalize(0.05, 0.12)
    ])
    strong_transform = transforms.Compose([
        transforms.Resize(size=(224,224), interpolation=InterpolationMode.LANCZOS),  # 32, 32
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(15),
        TargetedStrongAugment(n=4, window_size=112),  # 48
        transforms.ToTensor(),
        # transforms.Normalize(0.061, 0.140)
    ])
    labeled_dataset = datasets.ImageFolder(labeled_data_path, transform=transform_labeledtrain)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=16, shuffle=True, num_workers=4,
                                    drop_last=True)  # ,shuffle=True,pin_memory=True
    # labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.batch_size,sampler=train_sampler(labeled_dataset),num_workers=args.num_workers,drop_last=True)

    # 加载无标签数据集...
    unlabeled_dataset = datasets.ImageFolder(unlabeled_data_path,
                                             transform=TransformTwice(weak_transform, strong_transform))
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=(16) * (2), shuffle=True,
                                      num_workers=4,
                                      drop_last=True)  # unlabeled_dataset = [( (out1_weak_transform, out2_strong_transform), -1 )...shuffle=True,
    # unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=(args.batch_size)*(args.ulb_ratio),sampler=train_sampler(unlabeled_dataset),num_workers=args.num_workers,drop_last=True)

    val_dataset = datasets.ImageFolder(val_data_path, transform=transforms.Compose(
        [  #transforms.Resize(size=(112, 112), interpolation=InterpolationMode.LANCZOS),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), ]))
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4,
                                drop_last=False)

    device = torch.device("cuda")
    from torch_lr_finder import LRFinder
    model = fastvit_t12()
    model=LWN()
    model.to(device)
    #model.eval()
    #model=fast_vit()
    inp = torch.randn(2, 3, 112, 112).to(device)
    flops = FlopCountAnalysis(model, inp)
    print(flop_count_table(flops, max_depth=5))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(labeled_dataloader, val_loader=val_dataloader,start_lr=1e-6,end_lr=1, num_iter=100,step_mode="exp")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()
    # lr_finder.range_test(labeled_dataloader, start_lr=1e-6,end_lr=1e-2, num_iter=100, step_mode="exp")
    # lr_finder.plot(log_lr=False)
    # lr_finder.reset()
