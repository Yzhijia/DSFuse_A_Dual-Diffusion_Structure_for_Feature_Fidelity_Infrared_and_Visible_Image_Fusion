import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.utils as KU
from model.deform_conv_v2 import DeformConv2d

class Conv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, dilation=1, norm=None, act=nn.LeakyReLU,bias=False, deformable=False):
        super(Conv2d, self).__init__()
        model = []
        if deformable==True:
            model += [DeformConv2d(n_in, n_out, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, modulation=True)]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        if act is nn.LeakyReLU:
            model += [act(negative_slope=0.1,inplace=True)]
        elif act is None:
            model +=[]
        else:
            model +=[act()]
        self.model = nn.Sequential(*model)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)

class ResConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, dilation=1, norm=None,deformable=False):
        super(ResConv2d, self).__init__()
        model = []
        if deformable==True:
            model += [DeformConv2d(n_in, n_out, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False, modulation=True)]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False, dilation=dilation)]
        # model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
        #                     stride=stride, padding=padding, bias=False, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)+x