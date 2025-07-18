from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from .loss_ssim import ssim
import torchvision.transforms.functional as TF
from utils.hparams import hparams

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

class Intensity(nn.Module):
    def __init__(self):
        super(Intensity, self).__init__()

    def forward(self, image, image_fused):
        # intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, image)
        return Loss_intensity



class loss_SwinFusion(nn.Module):
    def __init__(self):
        super(loss_SwinFusion, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)

    def forward(self, image_A, image_B, image_fused):
        # loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        # loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        # loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))

        loss_l1 = hparams['p1'] * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = hparams['p2'] * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = hparams['p3'] * (1 - self.L_SSIM(image_A, image_B, image_fused))


        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM


class loss_SwinFusion_split(nn.Module):
    def __init__(self):
        super(loss_SwinFusion_split, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.Inten = Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)

    def forward(self, image_A, image_B, image_fused, image_Ar, image_Br):
        # loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        # loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        # loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))

        loss_l1 = hparams['p1'] * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = hparams['p2'] * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = hparams['p3'] * (1 - self.L_SSIM(image_A, image_B, image_fused))
        loss_int = hparams['p4'] * (self.Inten(image_A, image_Ar) + self.Inten(image_B, image_Br))

        fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_int
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM, loss_int