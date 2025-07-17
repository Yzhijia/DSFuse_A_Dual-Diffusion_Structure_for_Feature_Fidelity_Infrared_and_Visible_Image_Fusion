import functools
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from utils.hparams import hparams
from .module_util import make_layer, initialize_weights
from .commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from .commons import ResnetBlock, Upsample, Block, Downsample
from .layer import Conv2d, ResConv2d
import kornia.utils as KU
from .trans import *

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # if hparams['sr_scale'] == 8:
        #     self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        feas = []
        x = (x + 1) / 2
        fea_first = fea = self.conv_first(x)
        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)
        fea = self.lrelu(self.upconv1(fea))
        fea = self.lrelu(self.upconv2(fea))
        # if hparams['sr_scale'] == 8:
        #     fea = self.lrelu(self.upconv3(fea))
        # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # if hparams['sr_scale'] == 8:
        #     fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        out = out * 2 - 1
        if get_fea:
            return out, feas
        else:
            return out

# 固定参数

# rrdb_num_block 8
# sr_scale 8
# use_attn false
# res true
# up_input false
# use_wn false
# weight_init false


class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=3):
        super().__init__()
        dims = [1, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print("in_out"+str(in_out))
        groups = 0

        # self.cond_proj = nn.ConvTranspose2d(cond_dim * ((hparams['rrdb_num_block'] + 1) // 3),
        #                                     dim, hparams['sr_scale'] * 2, hparams['sr_scale'],
        #                                     hparams['sr_scale'] // 2)

        # self.cond_proj = nn.Conv2d(cond_dim * 3, dim, 3, 1, 1)

        # self.DM = DenseMatcher()
        # self.ST = SpatialTransformer(480, 640, True)



        # self.cond_proj = nn.Conv2d(1, dim, 3, 1, 1)

        # stn 网络

        # self.STN = STNMatcher()
        self.cond_proj = nn.Conv2d(cond_dim, dim, 3, 1, 1)
        # self.cond_proj = ResnetBlock(cond_dim, dim, time_emb_dim=0, groups=groups)
        # 仅配准的流程
        # self.cond_proj = nn.Conv2d(cond_dim, dim, 3, 1, 1)


        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.projs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity(),
                # ResnetBlock(dim_in, dim_out)
                # nn.Conv2d(dim_in, dim_out, 3, 1, 1)
            ]))
            # self.projs.append(nn.ModuleList([nn.Conv2d(dim_in, dim_out, 3, 1, 1)]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        if hparams['use_attn']:
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

        if hparams['res'] and hparams['up_input']:
            self.up_proj = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(3, dim, 3),
            )
        if hparams['use_wn']:
            self.apply_weight_norm()
        if hparams['weight_init']:
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x, time, cond):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []

        # ir = cond["ir"]
        # vis = cond["vis"]
        #
        # disp = self.DM(ir, vis)['ir2vis']
        # print(disp)
        # ir_reg = self.ST(ir, disp)
        # print(cond)
        # cond = self.cond_proj(torch.cat(cond, 1))
        # cond = self.cond_proj(torch.cat([ir, vis], 1).float())
        # print("cond:" + str(cond.shape))
        # cond = self.cond_proj(cond)
        cond = self.cond_proj(cond)
        # STN 网络
        # features = self.STN(ir, vis)
        # print(features.shape)
        # cond = self.cond_proj(features)


        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            # print(i)
            # print(x.shape)
            x = resnet(x, t)
            x = resnet2(x, t)

            if i == 0:
                # print("x:"+str(x.shape))
                # print("cond:" + str(cond.shape))

                x = x+cond
            # if i == 0:
            #     x = x + cond
            # else:
            #     x = x +
                # if hparams['res'] and hparams['up_input']:
                #     x = x + self.up_proj(img_lr_up)
            h.append(x)
            # cond = conv1(cond)
            # cond = downsample(cond)
            x = downsample(x)

        x = self.mid_block1(x, t)
        if hparams['use_attn']:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            #
            # a = h.pop()
            # print(a.shape)
            # print(x.shape)
            # x = torch.cat((x, a), dim=1)

            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)


    def forward2(self, x, time, cond, img_lr_up):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        cond = self.cond_proj(torch.cat(cond[2::3], 1))
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            print(i)
            x = resnet(x, t)
            x = resnet2(x, t)
            if i == 0:
                x = x + cond
                if hparams['res'] and hparams['up_input']:
                    x = x + self.up_proj(img_lr_up)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        if hparams['use_attn']:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)























# class encode_fusion_twochannel(nn.Module):
#     def __init__(self, depth):
#         super(encode_fusion_twochannel, self).__init__()
#         self.depth = depth
#
#


# sample:
class encode_fusion(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 3):
        super(encode_fusion, self).__init__()
        self.depth = depth

        self.encoder = nn.Sequential(
            nn.Conv2d(2, dim, (3, 3), (1, 1), 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.dense = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * (i + 1), dim, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ) for i in range(depth)
        ])

        self.fuse = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(dim * (depth + 1), dim * 4, (3, 3), (1, 1), 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim * 4, dim * 2, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim * 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim, 1, (3, 3), (1, 1), 1),
                nn.Tanh()
            ),
        )

    def forward(self, ir: Tensor, vi: Tensor, get_fea=False) -> Tensor:
        feas=[]
        src = torch.cat([ir, vi], dim=1)
        x = self.encoder(src.float())
        for i in range(self.depth):
            t = self.dense[i](x)
            x = torch.cat([x, t], dim=1)
            feas.append(t)
        fus = self.fuse(x)

        if get_fea:
            return fus, feas
        else:
            return fus



