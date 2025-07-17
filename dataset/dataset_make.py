import os.path

import math
import torch.optim as optim
from torch import nn
import numpy as np
# from scipy.misc import imread, imsave, imresize
import cv2
from os import listdir
from os.path import join
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
from dataset.imagecrop import FusionRandomCrop
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.autograd import Variable
import itertools
from utils.indexed_datasets import IndexedDataset
import random

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        FusionRandomCrop(crop_size),
    ])

def train_vis_ir_transform():
    return Compose([
		Grayscale(num_output_channels=1),
        ToTensor(),
    ])
def train_vis_transform():
    return Compose([
		# Grayscale(num_output_channels=1),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])
class Fusion_Dataset(Dataset):
    """
    root:数据集根目录
    augment:是否需要数据增强
    """
    # 获取数据集中所有图片的路径
    def __init__(self, directory_ir, directory_vi, directory_mask, crop_size, upscale_factor, prefix='train'):
        super(Fusion_Dataset, self).__init__()
        self.prefix = prefix
        self.image_lists_ir = []
        self.image_lists_vi = []
        self.image_lists_mask = []
        self.names = []
        self.item_name = []
        dir_ir = listdir(directory_ir)
        self.patch_size = 160
        dir_ir.sort()
        for file in dir_ir:
            name = file.lower()
            self.item_name.append(os.path.basename(file))
            if name.endswith('.png'):
                self.image_lists_ir.append(join(directory_ir, file))
                self.image_lists_vi.append(join(directory_vi, file))
                self.image_lists_mask.append(join(directory_mask, file))
            elif name.endswith('.jpg'):
                self.image_lists_ir.append(join(directory_ir, file))
                self.image_lists_vi.append(join(directory_vi, file))
                self.image_lists_mask.append(join(directory_mask, file))
            elif name.endswith('.jpeg'):
                self.image_lists_ir.append(join(directory_ir, file))
                self.image_lists_vi.append(join(directory_vi, file))
                self.image_lists_mask.append(join(directory_mask, file))
            elif name.endswith('.bmp'):
                self.image_lists_ir.append(join(directory_ir, file))
                self.image_lists_vi.append(join(directory_vi, file))
                self.image_lists_mask.append(join(directory_mask, file))
            elif name.endswith('.tif'):
                self.image_lists_ir.append(join(directory_ir, file))
                self.image_lists_vi.append(join(directory_vi, file))
                self.image_lists_mask.append(join(directory_mask, file))
            name1 = name.split('.')
            self.names.append(name1[0])
        self.data_len = len(self.image_lists_ir)

        # 计算crop_size
        if self.prefix == 'train':
            crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
            self.crop_transform = train_hr_transform(crop_size)
            self.vis_ir_transform = train_vis_ir_transform()
            self.ir_transform = train_vis_ir_transform()
            self.vis_transform = train_vis_transform()
            self.lr_transform = train_lr_transform(crop_size, upscale_factor)


    # def _get_item(self, index):
    #     if self.indexed_ds is None:
    #         self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
    #     return self.indexed_ds[index]



    def __getitem__(self, index):
        # 读取图像数据并返回





        if self.prefix == 'train':
            # print("********************")
            # try:
            visible_image = Image.open(self.image_lists_vi[index])
            infrared_image = Image.open(self.image_lists_ir[index])
            mask_image = Image.open(self.image_lists_mask[index])

            visible_image = visible_image.resize((480, 480))
            infrared_image = infrared_image.resize((480, 480))
            mask_image = mask_image.resize((480, 480))

            # try:
            crop_size = self.crop_transform(visible_image)
            # except Exception:
            #     print(index)
            #     print(self.image_lists_vi[index])
            visible_image, infrared_image, mask_image = F.crop(visible_image,crop_size[0],crop_size[1],crop_size[2],crop_size[3])\
                , F.crop(infrared_image, crop_size[0],crop_size[1],crop_size[2],crop_size[3]), F.crop(mask_image, crop_size[0],crop_size[1],crop_size[2],crop_size[3])

            visible_image = self.vis_transform(visible_image).double()
            infrared_image = self.ir_transform(infrared_image).double()
            mask_image = self.vis_ir_transform(mask_image).double()

            if (visible_image.shape[1] != infrared_image.shape[1])|(visible_image.shape[2] != infrared_image.shape[2]):
                print(self.image_lists_vi[index])
                print(visible_image.shape)
                print(infrared_image.shape)


            data = torch.cat(
                (self.lr_transform(infrared_image)[0].unsqueeze(0), self.lr_transform(visible_image)[0].unsqueeze(0)))
            # except Exception:
            #     print(self.image_lists_vi[index])
        elif self.prefix == 'test_full_scale':
            single_image_ir_path = self.image_lists_ir[index]
            single_image_vi_path = self.image_lists_vi[index]

            infrared_image = cv2.imread(single_image_ir_path, cv2.IMREAD_GRAYSCALE)
            visible_image = cv2.imread(single_image_vi_path, cv2.IMREAD_COLOR)
            mask_image = cv2.ximgproc.l0Smooth(infrared_image)

            visible_image = cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB)

            infrared_image = cv2.resize(infrared_image, (640, 480))
            visible_image = cv2.resize(visible_image, (640, 480))
            mask_image = cv2.resize(mask_image, (640, 480))

            infrared_image = infrared_image / 255
            visible_image = visible_image / 255
            mask_image = mask_image / 255

            infrared_image = np.reshape(infrared_image, [1, infrared_image.shape[0], infrared_image.shape[1]])
            # visible_image = np.reshape(visible_image, [3, visible_image.shape[0], visible_image.shape[1]])
            visible_image = visible_image.transpose(2, 0, 1)
            mask_image = np.reshape(mask_image, [1, mask_image.shape[0], mask_image.shape[1]])

        else:
            single_image_ir_path = self.image_lists_ir[index]
            single_image_vi_path = self.image_lists_vi[index]
            single_image_mask_path = self.image_lists_mask[index]

            # infrared_image = imread(single_image_ir_path, mode="L")
            # visible_image = imread(single_image_vi_path, mode="L")
            # mask_image = imread(single_image_mask_path, mode="L")


            infrared_image = cv2.imread(single_image_ir_path,cv2.IMREAD_GRAYSCALE)
            visible_image = cv2.imread(single_image_vi_path, cv2.IMREAD_COLOR)
            # mask_image = cv2.ximgproc.l0Smooth(infrared_image)
            mask_image = cv2.imread(single_image_mask_path,cv2.IMREAD_GRAYSCALE)

            visible_image = cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB)

            infrared_image=cv2.resize(infrared_image,(360,360))
            visible_image = cv2.resize(visible_image, (360,360))
            mask_image = cv2.resize(mask_image, (360,360))
            # infrared_image = cv2.resize(infrared_image, (160, 160))
            # visible_image = cv2.resize(visible_image, (160, 160))
            # mask_image = cv2.resize(mask_image, (160, 160))

            H, W = infrared_image.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            infrared_image = infrared_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            visible_image = visible_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            mask_image = mask_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]



            # 扩大黑边，保证随机矩阵变换完之后，没有图像内容的损失
            # infrared_image = cv2.copyMakeBorder(infrared_image, 60, 60, 80, 80, cv2.BORDER_CONSTANT, value=0)
            # visible_image = cv2.copyMakeBorder(visible_image, 60, 60, 80, 80, cv2.BORDER_CONSTANT, value=0)
            # mask_image = cv2.copyMakeBorder(mask_image, 60, 60, 80, 80, cv2.BORDER_CONSTANT, value=0)

            # print(infrared_image.shape)

            infrared_image = infrared_image / 255
            visible_image = visible_image / 255
            mask_image = mask_image / 255

            infrared_image = np.reshape(infrared_image, [1, infrared_image.shape[0], infrared_image.shape[1]])
            # visible_image = np.reshape(visible_image, [3, visible_image.shape[0], visible_image.shape[1]])
            visible_image = visible_image.transpose(2, 0, 1)
            mask_image = np.reshape(mask_image, [1, mask_image.shape[0], mask_image.shape[1]])

            # infrared_image = np.reshape(infrared_image, [1, infrared_image.shape[0], infrared_image.shape[1]])
            # visible_image = np.reshape(visible_image, [1, visible_image.shape[0], visible_image.shape[1]])
            # mask_image = np.reshape(mask_image, [1, mask_image.shape[0], mask_image.shape[1]])

            # data = torch.cat(
            #     (torch.from_numpy(infrared_image).float(), torch.from_numpy(visible_image).float()))

        # return data, infrared_image, visible_image, mask_image
        # return {'img_ir':infrared_image, 'img_vi':visible_image, 'img_mask':mask_image, 'img_cat':data, 'item_name':self.item_name[index]}
        return {'img_ir': np.float32(infrared_image), 'img_vi': np.float32(visible_image), 'item_name': self.item_name[index], 'mask_img':np.float32(mask_image)}

    def __len__(self):
        return self.data_len

