import math
import torch.optim as optim
from torch import nn
import numpy as np
# from scipy.misc import imread, imsave, imresize
from os import listdir
from os.path import join
import torch
import sys
sys.path.append(r"/home/yzj/code/transfusion_diffusion/")
from utils.hparams import hparams
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale

from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.autograd import Variable
import pytorch_msssim
import itertools
import cv2
import os

class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss,self).__init__()
        filter1_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float()
        filter2_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float()
        filter1_kernel = filter1_kernel.view(1, 1, 3, 3)
        filter2_kernel = filter2_kernel.view(1, 1, 3, 3)

        self.filter1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.filter1.weight.data = filter1_kernel
        self.filter2.weight.data = filter2_kernel

        self.filter1.weight.requires_grad = False
        self.filter2.weight.requires_grad = False

    def forward(self, x):

        x_grad = self.filter1(x)
        y_grad = self.filter2(x)



        out = torch.abs(x_grad)+torch.abs(y_grad)

        return out


class mask_pixel_loss(nn.Module):
    def __init__(self):
        super(mask_pixel_loss,self).__init__()

    # mask_image: mask
    # image1:fusion_image
    # image2:vis_image or ir_image
    def forward(self, mask_image, image1, image2):
        weight = image1.shape[2] * image2.shape[3]

        # pixel_loss = torch.pow(torch.norm((torch.mul(mask_image, image1 - image2))) , 2) / weight
        pixel_loss = torch.norm((torch.mul(mask_image, image1 - image2))) / weight
        return pixel_loss

def YCbCr2RGB(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128 / 255)
    G = Y - 0.34414 * (Cb - 128 / 255) - 0.71414 * (Cr - 128 / 255)
    B = Y + 1.772 * (Cb - 128 / 255)
    img_rgb = torch.cat((R, G, B),dim=1)
    return img_rgb

def RGB2YCbCr(R, G, B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    img_YCbCr = torch.cat((Y, Cb, Cr),dim=1)
    return img_YCbCr

def RGB2Y(R,G,B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y

def angle(a,b):
    vector = torch.multiply(a,b)
    up =torch.sum(vector)
    down = torch.sqrt(torch.sum(torch.square(a)))*torch.sqrt(torch.sum(torch.square(b)))
    # print("down:"+str(down))
    # print("up:" + str(up))
    theta = torch.acos(up/down)
    # theta = torch.abs(down-up)
    return theta


# class FusionLoss(nn.Module):
#     def __init__(self):
#         super(FusionLoss,self).__init__()
#         self.gradient_loss = gradientloss()
#         # self.gradientdirection_loss = gradientdirection_loss()
#         self.pixel_loss = mask_pixel_loss()
#         # self.pool = nn.MaxPool2d(2, 2)
#         self.ssim_loss = pytorch_msssim.msssim
#         self.pool = nn.AvgPool2d(2, 2)
#         print("参数：2")

#     def forward(self, ir_images, vi_images, fusion_images, ir_mask, vi_mask):

#         # 生成多尺度的红外，可见光以及融合后的图像图
#         ir_scale0 = ir_images

#         # visible_YCbCr = RGB2YCbCr(vi_images[:, 0, :, :].unsqueeze(1), vi_images[:, 1, :, :].unsqueeze(1),
#         #                           vi_images[:, 2, :, :].unsqueeze(1))

#         img_vi_gray = RGB2Y(vi_images[:, 0, :, :].unsqueeze(1), vi_images[:, 1, :, :].unsqueeze(1),
#                             vi_images[:, 2, :, :].unsqueeze(1))

#         vi_scale0 = img_vi_gray


#         fusion_scale0 = fusion_images






#         # fusion_color = YCbCr2RGB(fusion_images, visible_YCbCr[:, 1, :, :].unsqueeze(1),
#         #                          visible_YCbCr[:, 2, :, :].unsqueeze(1))



#         # 多尺度下的像素损失
#         ir_pixel_loss_scale0 = self.pixel_loss(ir_mask, fusion_scale0, ir_scale0)
#         vi_pixel_loss_scale0 = self.pixel_loss(vi_mask, fusion_scale0, vi_scale0)

#         # color_term = torch.mean(angle(vi_images[:, 0, :, :], fusion_color[:, 0, :, :]) + angle(vi_images[:, 1, :, :],fusion_color[:, 1, :,:]) + angle(vi_images[:, 2, :, :], fusion_color[:, 2, :, :]))
#         color_term=(1-self.ssim_loss(fusion_scale0,ir_scale0,normalize=True))+(1-self.ssim_loss(fusion_scale0,vi_scale0, normalize=True))
#         #

#         vi_grad_loss_scale0 = self.pixel_loss(vi_mask, self.gradient_loss(fusion_scale0), self.gradient_loss(vi_scale0))
#         ir_grad_loss_scale0 = self.pixel_loss(ir_mask, self.gradient_loss(fusion_scale0), self.gradient_loss(ir_scale0))
#         dif_grad_loss_scale0 = self.pixel_loss(1,self.gradient_loss(fusion_scale0),torch.max(self.gradient_loss(vi_scale0),self.gradient_loss(ir_scale0)))
#         total_loss = ir_pixel_loss_scale0 * hparams["p1"] + vi_pixel_loss_scale0 * hparams["p2"] + ir_grad_loss_scale0 * hparams["p3"] + vi_grad_loss_scale0 * hparams["p4"] + hparams["p5"] * dif_grad_loss_scale0+hparams["p6"]*color_term
#         return total_loss, ir_pixel_loss_scale0, vi_pixel_loss_scale0, ir_grad_loss_scale0, vi_grad_loss_scale0, color_term

class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss,self).__init__()
        self.gradient_loss = gradientloss()
        self.pixel_loss = mask_pixel_loss()
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)
        self.ssim_loss = pytorch_msssim.msssim
        print("****************************using Fusion Loss*******************************")

    def forward(self, ir_images, vi_images, fusion_images, ir_mask, vi_mask):

        # 生成多尺度的红外，可见光以及融合后的图像图
        ir_scale0 = ir_images


        vi_scale0 = torch.mean(vi_images,dim=1,keepdim=True)



        fusion_scale0 = torch.mean(fusion_images,dim=1,keepdim=True)


        # ir_mask_scale1 = self.pool(ir_mask)
        # vi_mask_scale1 = self.pool(vi_mask)

        # ir_mask=1
        # vi_mask=1


        # 多尺度下的像素损失
        ir_pixel_loss_scale0 = self.pixel_loss(ir_mask, fusion_scale0, ir_scale0)
        vi_pixel_loss_scale0 = self.pixel_loss(vi_mask, fusion_scale0, vi_scale0)


        # Average gradient
        # AG_loss = torch.sum(self.gradient_loss(fusion_scale0)) / (fusion_scale0.shape[2] * fusion_scale0.shape[3])
        # SD_loss = torch.pow(torch.norm(fusion_scale0 - torch.mean(fusion_scale0)) , 2) / (fusion_scale0.shape[2] * fusion_scale0.shape[3])
        #
        #
        #
        # vi_ssim_loss = 1 - self.ssim_loss(fusion_scale0,vi_scale0, normalize=True)
        # ir_ssim_loss = 1 - self.ssim_loss(fusion_scale0,ir_scale0, normalize=True)



        # 将像素损失更改成结构相似度损失
        # ir_pixel_loss_scale0 = 1 - self.ssim_loss(fusion_scale0,ir_scale0, normalize=True)
        # vi_pixel_loss_scale0 = 1 - self.ssim_loss(fusion_scale0,vi_scale0*(1-ir_mask), normalize=True)

        # pixel_loss_ir_vi = self.pixel_loss(1, fusion_scale0, torch.max(ir_scale0,vi_scale0))


        # ir_pixel_loss_scale2 = self.pixel_loss(ir_mask, fusion_scale2, ir_scale2)
        # ir_pixel_loss_scale3 = self.pixel_loss(ir_mask, fusion_scale3, ir_scale3)
        #
        # vi_pixel_loss_scale0 = self.pixel_loss(vi_mask, fusion_scale0, vi_scale0)
        # vi_pixel_loss_scale1 = self.pixel_loss(vi_mask, fusion_scale1, vi_scale1)
        # a = self.gradient_loss(fusion_scale0)
        # 多尺度下的梯度损失
        # a = self.gradient_loss(ir_scale0)

        # color_term = torch.mean(angle(vi_images[:,0,:,:],fusion_images[:,0,:,:])+angle(vi_images[:,1,:,:],fusion_images[:,1,:,:])+angle(vi_images[:,2,:,:],fusion_images[:,2,:,:]))


        vi_grad_loss_scale0 = self.pixel_loss(vi_mask, self.gradient_loss(fusion_scale0), self.gradient_loss(vi_scale0))
        ir_grad_loss_scale0 = self.pixel_loss(ir_mask, self.gradient_loss(fusion_scale0), self.gradient_loss(ir_scale0))
        # grad_loss = self.pixel_loss(1, self.gradient_loss(fusion_scale0),
        #                             (self.gradient_loss(vi_scale0)+self.gradient_loss(ir_scale0))/2)
        # grad_loss_ir_vi = self.pixel_loss(1,self.gradient_loss(fusion_scale0),torch.max(self.gradient_loss(vi_scale0),self.gradient_loss(ir_scale0)))
        dif_grad_loss_scale0 = self.pixel_loss(1,self.gradient_loss(fusion_scale0),torch.max(self.gradient_loss(vi_scale0),self.gradient_loss(ir_scale0)))
        # ir_grad_loss_scale0, vi_grad_loss_scale0 = self.gradientdirection_loss(ir_scale0, vi_scale0, fusion_scale0, ir_mask, vi_mask)
        # vi_grad_loss_scale0 = self.pixel_loss(vi_mask, self.gradient_loss(fusion_scale0), self.gradient_loss(vi_scale0))
        # vi_grad_loss_scale1 = self.pixel_loss(vi_mask, self.gradient_loss(fusion_scale1), self.gradient_loss(vi_scale1))
        #
        # ir_grad_loss_scale2 = self.pixel_loss(ir_mask, self.gradient_loss(fusion_scale2), self.gradient_loss(ir_scale2))
        # ir_grad_loss_scale3 = self.pixel_loss(ir_mask, self.gradient_loss(fusion_scale3), self.gradient_loss(ir_scale3))


        # 总体损失函数构建
        # total_loss = (ir_pixel_loss_scale2 + ir_pixel_loss_scale3)*7 + (vi_pixel_loss_scale0 + vi_pixel_loss_scale1)*5 + (ir_grad_loss_scale2 + ir_grad_loss_scale3)*10 + (vi_grad_loss_scale0 + vi_grad_loss_scale1)*10
        # total_loss = ir_pixel_loss_scale0 * 7 + vi_pixel_loss_scale0 * 5 + ir_grad_loss_scale0 * 10 + vi_grad_loss_scale0 * 20 + dif_grad_loss_scale0 * 10
        total_loss = ir_pixel_loss_scale0 * hparams["p1"] + vi_pixel_loss_scale0 * hparams["p2"] + ir_grad_loss_scale0 * hparams["p3"] + vi_grad_loss_scale0 * hparams["p4"] + hparams["p5"] * dif_grad_loss_scale0
        # total_loss = ir_pixel_loss_scale0 * 7 + vi_pixel_loss_scale0 * 7 + grad_loss * 20
        # total_loss =  ir_pixel_loss_scale0 * 7 + vi_pixel_loss_scale0 * 7 + grad_loss_ir_vi * 5
        # total_loss = ir_pixel_loss_scale0 * 7 + vi_pixel_loss_scale0 * 5 + vi_grad_loss_scale0 * 20
        # 返回值， total_loss, ir_pixel_loss, vi_pixel_loss, ir_grad_loss, vi_grad_loss
        # return total_loss, ir_pixel_loss_scale2 + ir_pixel_loss_scale3, vi_pixel_loss_scale0 + vi_pixel_loss_scale1, ir_grad_loss_scale2 + ir_grad_loss_scale3, vi_grad_loss_scale0 + vi_grad_loss_scale1
        return total_loss, ir_pixel_loss_scale0, vi_pixel_loss_scale0, ir_grad_loss_scale0, vi_grad_loss_scale0