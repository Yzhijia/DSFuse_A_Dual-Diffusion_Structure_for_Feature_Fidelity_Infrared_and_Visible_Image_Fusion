import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .layer import *
import kornia.utils as KU
import kornia.filters as KF
from copy import deepcopy
import os
import numpy as np
from PIL import Image
import cv2
from model.deform_conv_v2 import DeformConv2d


class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1]==2:
            disp = disp.permute(0,2,3,1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1],disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow.double(), mode=self.mode, padding_mode='zeros', align_corners=False)


class DispEstimator(nn.Module):
    def __init__(self,channel,depth=4,norm=nn.BatchNorm2d,dilation=1):
        super(DispEstimator,self).__init__()
        estimator = nn.ModuleList([])
        self.corrks = 7
        self.preprocessor = Conv2d(channel,channel,3,act=None,norm=None,dilation=dilation,padding=dilation)
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,act=None))
        #self.localcorrpropcessor = nn.Sequential(Conv2d(self.corrks**2,32,3,padding=1,bias=True,norm=None),
        #                                         Conv2d(32,2,3,padding=1,bias=True,norm=None),)
        oc = channel
        ic = channel+self.corrks**2
        dilation = 1
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.layers = estimator
        self.scale = torch.FloatTensor([256,256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
        #self.corrpropcessor = Conv2d(9+channel,channel,3,padding=1,bias=True,norm=nn.InstanceNorm2d)
        #self.AP3=nn.AvgPool2d(3,stride=1,padding=1)

    # def localcorr(self,feat1,feat2):
    #     feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
    #     feat1 = F.normalize(feat1,dim=1)
    #     feat2 = F.normalize(feat2,dim=1)
    #     b,c,h,w = feat2.shape
    #     feat2_smooth = KF.gaussian_blur2d(feat2,[9,9],[3,3])
    #     feat2_loc_blk = F.unfold(feat2_smooth,kernel_size=self.corrks,dilation=4,padding=4*(self.corrks-1)//2,stride=1).reshape(b,c,-1,h,w)
    #     localcorr = (feat1.unsqueeze(2)*feat2_loc_blk).sum(dim=1)
    #     localcorr = self.localcorrpropcessor(localcorr)
    #     corr = torch.cat([feat,localcorr],dim=1)
    #     return corr
    def localcorr(self,feat1,feat2):
        feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
        b,c,h,w = feat2.shape
        feat1_smooth = KF.gaussian_blur2d(feat1,(13,13),(3,3),border_type='constant')
        feat1_loc_blk = F.unfold(feat1_smooth,kernel_size=self.corrks,dilation=4,padding=2*(self.corrks-1),stride=1).reshape(b,c,-1,h,w)
        localcorr = (feat2.unsqueeze(2)-feat1_loc_blk).pow(2).mean(dim=1)
        corr = torch.cat([feat,localcorr],dim=1)
        return corr

    def forward(self,feat1,feat2):
        b,c,h,w = feat1.shape
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat1 = feat[:b]
        feat2 = feat[b:]
        if self.scale[0,1,0,0] != w-1 or self.scale[0,0,0,0] != h-1:
            self.scale = torch.FloatTensor([w,h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
            self.scale = self.scale.to(feat1.device)
        corr = self.localcorr(feat1,feat2)
        for i,layer in enumerate(self.layers):
            corr = layer(corr)
        corr = KF.gaussian_blur2d(corr,(13,13),(3,3),border_type='replicate')
        disp = corr.clamp(min=-300,max=300)
        # print(disp.shape)
        # print(feat1.shape)
        return disp/self.scale


class DispRefiner(nn.Module):
    def __init__(self, channel, dilation=1, depth=4):
        super(DispRefiner, self).__init__()
        self.preprocessor = nn.Sequential(
            Conv2d(channel, channel, 3, dilation=dilation, padding=dilation, norm=None, act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel * 2, channel * 2, 3, padding=1),
                                            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=None))
        oc = channel
        ic = channel + 2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth - 1):
            oc = oc // 2
            estimator.append(
                Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc, 2, kernel_size=3, padding=1, dilation=1, act=None, norm=None))
        # estimator.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator)

    def forward(self, feat1, feat2, disp):
        b = feat1.shape[0]
        feat = torch.cat([feat1, feat2])
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b], feat[b:]], dim=1))
        corr = torch.cat([feat, disp], dim=1)
        delta_disp = self.estimator(corr)
        disp = disp + delta_disp
        return disp


class Feature_extractor_unshare(nn.Module):
    def __init__(self,depth,base_ic,base_oc,base_dilation,norm, deformable=False):
        super(Feature_extractor_unshare,self).__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i%2==1:
                dilation *= 2

            # if deformable==True:
            #     feature_extractor.append(
            #         DeformConv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm))

            if ic == oc:
                feature_extractor.append(ResConv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm, deformable=deformable))
            else:
                feature_extractor.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm, deformable=deformable))
            ic = oc
            if i%2==1 and i<depth-1:
                oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x




class STNMatcher(nn.Module):
    def __init__(self, unshare_depth=4, matcher_depth=4, num_pyramids=2):
        super(STNMatcher, self).__init__()

        self.feature_extractor_unshare1 = Feature_extractor_unshare(depth=unshare_depth, base_ic=1, base_oc=8,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d, deformable=False)
        self.feature_extractor_unshare2 = Feature_extractor_unshare(depth=unshare_depth, base_ic=1, base_oc=8,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d)

        base_ic = self.feature_extractor_unshare1.ic
        base_oc = self.feature_extractor_unshare1.oc
        # print(base_oc)

        self.feature_extractor_share1 = nn.Sequential(
            Conv2d(base_oc, base_oc * 2, kernel_size=3, stride=1, padding=1, dilation=1, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 2, base_oc * 2, kernel_size=3, stride=1, padding=1, dilation=1, norm=nn.InstanceNorm2d))

        self.localization = nn.Sequential(
            Conv2d(base_oc * 2, base_oc, kernel_size=6, stride=2, padding=1, dilation=1, norm=nn.InstanceNorm2d),
            nn.MaxPool2d(2, stride=2),
            Conv2d(base_oc, 4, kernel_size=6, stride=2, padding=1, dilation=1, norm=nn.InstanceNorm2d),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(4524, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

    def stn(self, x):
        xs = self.localization(x)
        # print(xs.shape)
        xs = xs.view(-1, 4524)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # theta[0,0,0]=0.707
        # theta[0,0,1]=-0.707
        # theta[0,0,2]=0
        #
        # theta[0,1,0]=0.707
        # theta[0, 1, 1] = 0.707
        # theta[0, 1, 2] = 0
        # print(theta)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        # print(torch.max(x))
        return (x, grid)

    def forward(self, src, tgt):
        b, c, h, w = tgt.shape
        feat01 = self.feature_extractor_unshare1(src.float())
        feat02 = self.feature_extractor_unshare2(tgt.float())
        feat0 = torch.cat([feat01, feat02])
        feat1 = self.feature_extractor_share1(feat0)
        # print(feat1.shape)
        feat11, feat12 = feat1[0:b], feat1[b:]
        # (feat11_r, grid) = self.stn(feat11)
        # feat11_r = self.stn(feat11)
        # print(torch.max(feat11))

        # src_img = np.squeeze(tensor2img(src))
        # M = np.squeeze(theta.cpu().numpy())
        # print(M)
        # re = cv2.warpAffine(src_img, M, (4000, 4000))
        # cv2.imwrite(f"/home/yzj/code/fusion_diffusion/checkpoints/diff_fusion/results_0_/1.png", re)

        #
        # src_a = F.grid_sample(tgt.float(), grid)
        # print(torch.max(src_a.float()))
        # img = Image.fromarray(np.squeeze(tensor2img(src_a))).save(f"/home/yzj/code/fusion_diffusion/checkpoints/diff_fusion/results_0_/1.png")
        # src_a = F.grid_sample(src_a.float(), -grid)
        # img = Image.fromarray(np.squeeze(tensor2img(src_a))).save(
        #     f"/home/yzj/code/fusion_diffusion/checkpoints/diff_fusion/results_0_/2.png")
        # # print(img.unique())
        # print("图像保存成功！")





        return torch.cat([feat11, feat12], 1)
        # 仅仅配准的流程
        # return feat1







def tensor2img(img):
    # img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
    img = np.round((img.permute(0, 2, 3, 1).cpu().numpy()) * 255)
    img = img.clip(min=0, max=255).astype(np.uint8)
    return img


def tensor2grid(grid):
    grid=grid.permute(0, 2, 3, 1)
    return grid







class DenseMatcher(nn.Module):
    def __init__(self, unshare_depth=4, matcher_depth=4, num_pyramids=2):
        super(DenseMatcher, self).__init__()
        self.num_pyramids = num_pyramids
        self.feature_extractor_unshare1 = Feature_extractor_unshare(depth=unshare_depth, base_ic=1, base_oc=8,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d)
        self.feature_extractor_unshare2 = Feature_extractor_unshare(depth=unshare_depth, base_ic=1, base_oc=8,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d)
        # self.feature_extractor_unshare2 = self.feature_extractor_unshare1
        base_ic = self.feature_extractor_unshare1.ic
        base_oc = self.feature_extractor_unshare1.oc
        base_dilation = self.feature_extractor_unshare1.dilation
        self.feature_extractor_share1 = nn.Sequential(
            Conv2d(base_oc, base_oc * 2, kernel_size=3, stride=1, padding=1, dilation=1, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 2, base_oc * 2, kernel_size=3, stride=2, padding=1, dilation=1, norm=nn.InstanceNorm2d))
        self.feature_extractor_share2 = nn.Sequential(
            Conv2d(base_oc * 2, base_oc * 4, kernel_size=3, stride=1, padding=2, dilation=2, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 4, base_oc * 4, kernel_size=3, stride=2, padding=2, dilation=2, norm=nn.InstanceNorm2d))
        self.feature_extractor_share3 = nn.Sequential(
            Conv2d(base_oc * 4, base_oc * 8, kernel_size=3, stride=1, padding=4, dilation=4, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 8, base_oc * 8, kernel_size=3, stride=2, padding=4, dilation=4, norm=nn.InstanceNorm2d))
        self.matcher1 = DispEstimator(base_oc * 4, matcher_depth, dilation=4)
        self.matcher2 = DispEstimator(base_oc * 8, matcher_depth, dilation=2)
        self.refiner = DispRefiner(base_oc * 2, 1)
        self.grid_down = KU.create_meshgrid(64, 64).cuda()
        self.grid_full = KU.create_meshgrid(128, 128).cuda()
        self.scale = torch.FloatTensor([128, 128]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1

    def match(self, feat11, feat12, feat21, feat22, feat31, feat32):
        # compute scale (w,h)
        if self.scale[0, 1, 0, 0] * 2 != feat11.shape[2] - 1 or self.scale[0, 0, 0, 0] * 2 != feat11.shape[3] - 1:
            self.h, self.w = feat11.shape[2], feat11.shape[3]
            self.scale = torch.FloatTensor([self.w, self.h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
            self.scale = self.scale.to(feat11.device)

        # estimate disp src(feat1) to tgt(feat2) in low resolution
        disp2_raw = self.matcher2(feat31, feat32)

        # upsample disp and grid
        disp2 = F.interpolate(disp2_raw, [feat21.shape[2], feat21.shape[3]], mode='bilinear')
        if disp2.shape[2] != self.grid_down.shape[1] or disp2.shape[3] != self.grid_down.shape[2]:
            self.grid_down = KU.create_meshgrid(feat21.shape[2], feat21.shape[3]).cuda()

        # warp the last src(fea1) to tgt(feat2) with disp2
        feat21 = F.grid_sample(feat21, self.grid_down + disp2.permute(0, 2, 3, 1))

        # estimate disp src(feat1) to tgt(feat2) in low resolution
        disp1_raw = self.matcher1(feat21, feat22)

        # upsample
        disp1 = F.interpolate(disp1_raw, [feat11.shape[2], feat11.shape[3]], mode='bilinear')
        disp2 = F.interpolate(disp2, [feat11.shape[2], feat11.shape[3]], mode='bilinear')
        if disp1.shape[2] != self.grid_full.shape[1] or disp1.shape[3] != self.grid_full.shape[2]:
            self.grid_full = KU.create_meshgrid(feat11.shape[2], feat11.shape[3]).cuda()

        # warp
        feat11 = F.grid_sample(feat11, self.grid_full + (disp1 + disp2).permute(0, 2, 3, 1))

        # finetune
        disp_scaleup = (disp1 + disp2) * self.scale
        disp = self.refiner(feat11, feat12, disp_scaleup)
        disp = KF.gaussian_blur2d(disp, (17, 17), (5, 5), border_type='replicate') / self.scale
        if self.training:
            return disp, disp_scaleup / self.scale, disp2
        return disp, None, None

    def forward(self, src, tgt, type='ir2vis'):
        b, c, h, w = tgt.shape
        feat01 = self.feature_extractor_unshare1(src.float())
        feat02 = self.feature_extractor_unshare2(tgt.float())
        feat0 = torch.cat([feat01, feat02])
        feat1 = self.feature_extractor_share1(feat0)
        feat2 = self.feature_extractor_share2(feat1)
        feat3 = self.feature_extractor_share3(feat2)
        feat11, feat12 = feat1[0:b], feat1[b:]
        feat21, feat22 = feat2[0:b], feat2[b:]
        feat31, feat32 = feat3[0:b], feat3[b:]
        disp_12 = None
        disp_21 = None
        if type == 'bi':
            disp_12, disp_12_down4, disp_12_down8 = self.match(feat11, feat12, feat21, feat22, feat31, feat32)
            disp_21, disp_21_down4, disp_21_down8 = self.match(feat12, feat11, feat22, feat21, feat32, feat31)
            t = torch.cat([disp_12, disp_21, disp_12_down4, disp_21_down4, disp_12_down8, disp_21_down8])
            t = F.interpolate(t, [h, w], mode='bilinear')
            down2, down4, donw8 = torch.split(t, 2 * b, dim=0)
            disp_12_, disp_21_ = torch.split(down2, b, dim=0)
        elif type == 'ir2vis':
            disp_12, _, _ = self.match(feat11, feat12, feat21, feat22, feat31, feat32)
            disp_12 = F.interpolate(disp_12, [h, w], mode='bilinear')
        elif type == 'vis2ir':
            disp_21, _, _ = self.match(feat12, feat11, feat22, feat21, feat32, feat31)
            disp_21 = F.interpolate(disp_21, [h, w], mode='bilinear')
        # if self.training:
        #     return {'ir2vis': disp_12_, 'vis2ir': disp_21_,
        #             'down2': down2,
        #             'down4': down4,
        #             'down8': donw8}
        return {'ir2vis': disp_12, 'vis2ir': disp_21}


