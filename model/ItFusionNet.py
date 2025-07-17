import math
import torch.optim as optim
from torch import nn
import numpy as np
from os import listdir
from os.path import join
import torch
from torch.nn.utils.spectral_norm import spectral_norm
import math
# from commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
# from commons import ResnetBlock, Upsample, Block, Downsample
import torch
import torch.nn as nn
import torch.nn.init as init
from model.module_util import make_layer, initialize_weights
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.autograd import Variable
import functools


class ResnetBlock(nn.Module):
    def __init__(self, input_nc, output_nc, activate_func):
        super(ResnetBlock, self).__init__()
        self.block1 = nn.Sequential(

            nn.Conv2d(input_nc, input_nc, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_nc),
            nn.LeakyReLU(0.2),
            nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_nc),
            nn.LeakyReLU(0.2),
            nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0),
        )
        self.block1_identity_conv = nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(output_nc)
        if activate_func == "LeakyRelu":
            self.activate_func = nn.LeakyReLU(0.2)
        elif activate_func == "elu":
            self.activate_func = nn.ELU()
        elif activate_func == "tanh":
            self.activate_func = torch.tanh

    def forward(self, x):
        block1_output = self.block1(x)
        out = block1_output + self.block1_identity_conv(x)

        return out


# 增加instancenorm，归一化层
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.bn1(self.conv2d(out))
        #out = self.conv2d(out)
        if self.is_last is False:
            out = self.leakyrelu(out)
        return out

# Dense Block unit
# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, kernel_size=1, stride=stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class multi_scale_block(nn.Module):
    def __init__(self, in_channels, nb_filter):
        super(multi_scale_block,self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        block = DenseBlock_light
        kernel_size = 3
        stride = 1
        layer1_num = int(nb_filter[0] / 3)
        layer2_num = nb_filter[2] - nb_filter[3]
        layer3_num = nb_filter[3]
        layer4_num = layer3_num

        self.DB1_1 = block(in_channels, layer1_num, kernel_size=kernel_size, stride=stride)
        self.DB2_1 = block(layer1_num, layer2_num, kernel_size=kernel_size, stride=stride)
        self.DB3_1 = block(layer2_num, layer3_num, kernel_size=kernel_size, stride=stride)

        self.DB1_2 = block(layer1_num, layer1_num, kernel_size=kernel_size, stride=stride)
        self.DB2_2 = block(layer1_num + layer2_num, layer2_num, kernel_size=kernel_size, stride=stride)
        self.DB1_3 = block(layer1_num * 2, layer1_num, kernel_size=kernel_size, stride=stride)




    def forward(self, x):
        x1_1 = self.DB1_1(x)
        x1_2 = self.DB1_2(x1_1)
        x1_3 = self.DB1_3(torch.cat([x1_1, x1_2], 1))

        x2_1 = self.DB2_1(self.pool(x1_1))
        x2_2 = self.DB2_2(torch.cat([self.pool(x1_2), x2_1],1))

        x3_1 = self.DB3_1(self.pool(x2_1))


        out_features1 = torch.cat([x1_1, x1_2, x1_3], 1)
        out_features2 = torch.cat([x2_1, x2_2, self.pool(x1_3)], 1)
        out_features3 = torch.cat([x3_1, self.pool(x2_2)], 1)
        out_features4 = self.pool(x3_1)
        return [out_features1, out_features2, out_features3, out_features4]


class multi_scale_decoder(nn.Module):
    def __init__(self, nb_filter):
        super(multi_scale_decoder, self).__init__()
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        # # no short connection
        # self.DB1_2 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        # self.DB2_2 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        # self.DB1_3 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)

        # short connection
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

    def forward(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))


        return x1_3

# nb_filter = [96, 128, 96, 48]

class Feature_encoder(nn.Module):
    def __init__(self, nb_filter):
        super(Feature_encoder,self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.block1 = ResnetBlock(input_nc=16, output_nc=32, activate_func="LeakyRelu")
        self.block2 = multi_scale_block(in_channels=32, nb_filter=nb_filter)

    def forward(self, x):
        block1_input = self.leakyrelu(self.conv1(x))
        block1_output = self.block1(block1_input)
        out_features = self.block2(block1_output)

        return out_features




class Feature_decoder(nn.Module):
    def __init__(self, nb_filter):
        super(Feature_decoder,self).__init__()
        self.block1 = multi_scale_decoder(nb_filter=nb_filter)
        self.block2 = ResnetBlock(input_nc=96, output_nc=48, activate_func="LeakyRelu")
        self.block3 = ResnetBlock(input_nc=48, output_nc=24, activate_func="LeakyRelu")
        #self.block4 = ResnetBlock(input_nc=24, output_nc=1, activate_func="tanh")
        block4 = []
        block4.append(nn.Conv2d(24, 1, kernel_size=3, padding=1))  # amazing
        self.block4 = nn.Sequential(*block4)

    def forward(self,x):
        block1_output = self.block1(x)
        block2_output = self.block2(block1_output)
        block3_output = self.block3(block2_output)
        out = self.block4(block3_output)
        return (torch.tanh(out)+1)/2


class Feature_decoder_fusion(nn.Module):
    def __init__(self, nb_filter):
        super(Feature_decoder_fusion, self).__init__()
        self.block1 = multi_scale_decoder(nb_filter=nb_filter)
        self.block2 = ResnetBlock(input_nc=96 * 2, output_nc=48 * 2, activate_func="LeakyRelu")
        self.block3 = ResnetBlock(input_nc=48 * 2, output_nc=24 * 2, activate_func="LeakyRelu")
        self.block4 = ResnetBlock(input_nc=24 * 2, output_nc=24, activate_func="LeakyRelu")
        block5 = []
        block5.append(nn.Conv2d(24, 1, kernel_size=3, padding=1))  # amazing
        self.block5 = nn.Sequential(*block5)
        #self.block5 = ResnetBlock(input_nc=24, output_nc=1, activate_func="tanh")
    def forward(self, x):
        block1_output = self.block1(x)
        block2_output = self.block2(block1_output)
        block3_output = self.block3(block2_output)
        block4_output = self.block4(block3_output)
        out = self.block5(block4_output)
        return (torch.tanh(out)+1)/2




















class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


class Feature_Extraction_Net(nn.Module):
    def __init__(self):
        super(Feature_Extraction_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.block1 = ResnetBlock(input_nc=16, output_nc=16, activate_func="LeakyRelu")
        self.block2 = ResnetBlock(input_nc=16, output_nc=32, activate_func="LeakyRelu")
        self.block3 = ResnetBlock(input_nc=32, output_nc=64, activate_func="LeakyRelu")

    def forward(self,x):
        block1_input = self.leakyrelu(self.conv1(x))
        block1_output = self.block1(block1_input)
        block2_output = self.block2(block1_output)
        out_features = self.block3(block2_output)
        return out_features



class Feature_Reconstruction_Net(nn.Module):
    def __init__(self, inchannels):
        super(Feature_Reconstruction_Net, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.elu = nn.ELU()
        self.block1 = nn.Sequential(
            nn.Conv2d(inchannels, 128, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
        )
        self.block1_identity_conv = nn.Conv2d(inchannels, 64, kernel_size=1, stride=1, padding=0)

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
        )
        self.block2_identity_conv = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
        )
        self.block3_identity_conv = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)

        self.block4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),

        )
        self.block4_identity_conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        block1_output = self.block1(x)
        block2_input = self.elu(block1_output + self.block1_identity_conv(x))

        block2_output = self.block2(block2_input)
        block3_input = self.elu(block2_output + self.block2_identity_conv(block2_input))

        block3_output = self.block3(block3_input)
        # block4_input = self.leakyrelu(block3_output + self.block3_identity_conv(block3_input))
        block4_input = self.leakyrelu(block3_output + self.block3_identity_conv(block3_input))



        block4_output = self.block4(block4_input)
        out = torch.tanh(block4_output + self.block4_identity_conv(block4_input))

        return out



class Discriminator(nn.Module):
    def __init__(self, M=128):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            spectral_norm(nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(
                64, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            spectral_norm(nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(
                128, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            spectral_norm(nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(
                256, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            spectral_norm(nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),


            spectral_norm(nn.Conv2d(
                512, 16, kernel_size=1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = spectral_norm(
            nn.Linear(M // 8 * M // 8 * 16, 1, bias=False))
        res_arch_init(self)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x



def res_arch_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)




class AttentionBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, f_ca=1, f_sa=1, downsample=None):
        super(AttentionBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.f_ca = f_ca
        self.f_sa = f_sa

        self.downsample = downsample
        self.stride = stride

        # 对原始尺寸进行降维
        self.identity_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        residual = self.identity_conv(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.f_ca ==1 :
            out = self.ca(out) * out
        if self.f_sa ==1 :
            out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Multiscale_AttentionBlock(nn.Module):
    def __init__(self,nb_filter):
        super(Multiscale_AttentionBlock,self).__init__()


        self.scale0 = AttentionBlock(nb_filter[0],int(nb_filter[0]/2), f_sa=1, f_ca=1)
        self.scale1 = AttentionBlock(nb_filter[1],int(nb_filter[1]/2), f_sa=1, f_ca=1)
        self.scale2 = AttentionBlock(nb_filter[2],int(nb_filter[2]/2), f_sa=1, f_ca=1)
        self.scale3 = AttentionBlock(nb_filter[3],int(nb_filter[3]/2), f_sa=1, f_ca=1)

        # self.scale0 = AttentionBlock(nb_filter[0], nb_filter[0])
        # self.scale1 = AttentionBlock(nb_filter[1], nb_filter[1])
        # self.scale2 = AttentionBlock(nb_filter[2], nb_filter[2])
        # self.scale3 = AttentionBlock(nb_filter[3], nb_filter[3])


    def forward(self, f_en):
        out0 = self.scale0(f_en[0])
        out1 = self.scale1(f_en[1])
        out2 = self.scale2(f_en[2])
        out3 = self.scale3(f_en[3])

        return [out0, out1, out2, out3]



# 注意力机制
class FeatureAttentionNet(nn.Module):
    def __init__(self, nb_filter):
        super(FeatureAttentionNet,self).__init__()
        self.ir_attention = Multiscale_AttentionBlock(nb_filter=nb_filter)
        self.vi_attention = Multiscale_AttentionBlock(nb_filter=nb_filter)
        self.nb_filter = nb_filter

    def forward(self, ir_features, vi_features):
        ir_a_features = self.ir_attention(ir_features)
        vi_a_features = self.vi_attention(vi_features)

        out0 = torch.cat([ir_a_features[0], vi_a_features[0]], dim=1)
        out1 = torch.cat([ir_a_features[1], vi_a_features[1]], dim=1)
        out2 = torch.cat([ir_a_features[2], vi_a_features[2]], dim=1)
        out3 = torch.cat([ir_a_features[3], vi_a_features[3]], dim=1)

        return [out0, out1, out2, out3]


# 注意力机制
class FeatureAttentionNet_withmask(nn.Module):
    def __init__(self, nb_filter):
        super(FeatureAttentionNet_withmask, self).__init__()
        self.ir_attention = Multiscale_AttentionBlock(nb_filter=nb_filter)
        self.vi_attention = Multiscale_AttentionBlock(nb_filter=nb_filter)
        # self.attention = Multiscale_AttentionBlock(nb_filter=[i*2 for i in nb_filter])
        self.nb_filter = nb_filter
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, ir_features, vi_features):
        ir_a_features = self.ir_attention(ir_features)
        vi_a_features = self.vi_attention(vi_features)

        # feature0 = torch.cat([ir_features[0], vi_features[0]], dim=1)
        # feature1 = torch.cat([ir_features[1], vi_features[1]], dim=1)
        # feature2 = torch.cat([ir_features[2], vi_features[2]], dim=1)
        # feature3 = torch.cat([ir_features[3], vi_features[3]], dim=1)
        #
        # features = [feature0, feature1, feature2, feature3]
        #
        # a_features = self.attention(features)

        # # 红外掩膜尺寸变化
        #
        #
        # ir_mask_scale1 = self.pool(ir_mask)
        # ir_mask_scale2 = self.pool(ir_mask_scale1)
        # ir_mask_scale3 = self.pool(ir_mask_scale2)
        #
        # ir_mask_scale0 = ir_mask.repeat(1, int(self.nb_filter[0]/2), 1, 1)
        # ir_mask_scale1 = ir_mask_scale1.repeat(1, int(self.nb_filter[1]/2), 1, 1)
        # ir_mask_scale2 = ir_mask_scale2.repeat(1, int(self.nb_filter[2]/2), 1, 1)
        # ir_mask_scale3 = ir_mask_scale3.repeat(1, int(self.nb_filter[3]/2), 1, 1)
        #
        # #可见光掩膜尺寸变化
        # vi_mask_scale1 = self.pool(vi_mask)
        # vi_mask_scale2 = self.pool(vi_mask_scale1)
        # vi_mask_scale3 = self.pool(vi_mask_scale2)
        #
        # vi_mask_scale0 = vi_mask.repeat(1, int(self.nb_filter[0]/2), 1, 1)
        # vi_mask_scale1 = vi_mask_scale1.repeat(1, int(self.nb_filter[1]/2), 1, 1)
        # vi_mask_scale2 = vi_mask_scale2.repeat(1, int(self.nb_filter[2]/2), 1, 1)
        # vi_mask_scale3 = vi_mask_scale3.repeat(1, int(self.nb_filter[3]/2), 1, 1)


        # ir_a_features[0] = ir_a_features[0] * ir_mask_scale0
        # ir_a_features[1] = ir_a_features[1] * ir_mask_scale1
        # ir_a_features[2] = ir_a_features[2] * ir_mask_scale2
        # ir_a_features[3] = ir_a_features[3] * ir_mask_scale3
        #
        # vi_a_features[0] = vi_a_features[0] * vi_mask_scale0
        # vi_a_features[1] = vi_a_features[1] * vi_mask_scale1
        # vi_a_features[2] = vi_a_features[2] * vi_mask_scale2
        # vi_a_features[3] = vi_a_features[3] * vi_mask_scale3

        # ir_a_features[0] = ir_a_features[0]
        # ir_a_features[1] = ir_a_features[1]
        # ir_a_features[2] = ir_a_features[2]
        # ir_a_features[3] = ir_a_features[3]
        #
        # vi_a_features[0] = vi_a_features[0]
        # vi_a_features[1] = vi_a_features[1]
        # vi_a_features[2] = vi_a_features[2]
        # vi_a_features[3] = vi_a_features[3]


        out0 = torch.cat([ir_a_features[0], vi_a_features[0]], dim=1)
        out1 = torch.cat([ir_a_features[1], vi_a_features[1]], dim=1)
        out2 = torch.cat([ir_a_features[2], vi_a_features[2]], dim=1)
        out3 = torch.cat([ir_a_features[3], vi_a_features[3]], dim=1)

        return [out0, out1, out2, out3]
        # return a_features


class ItFusionNet(nn.Module):
    def __init__(self, encoder_ir_path="/home/yzj/code/transfusion_diffusion/fusion_weight/epoch149_ir_extraction.pth", 
                 encoder_vi_path="/home/yzj/code/transfusion_diffusion/fusion_weight/epoch149_vi_extraction.pth", 
                 attention_path="/home/yzj/code/transfusion_diffusion/fusion_weight/epoch45_net_attention.pth",
                 decoder_path="/home/yzj/code/transfusion_diffusion/fusion_weight/epoch45_Fusion_decoder.pth"):
        super(ItFusionNet, self).__init__()
        self.net_ir = Feature_encoder(nb_filter=[96, 128, 96, 48])
        self.net_vi = Feature_encoder(nb_filter=[96, 128, 96, 48])
        self.net_attention = FeatureAttentionNet_withmask(nb_filter=[96, 128, 96, 48])
        self.net_feature_decoder = Feature_decoder(nb_filter=[96, 128, 96, 48])

        # 载入参数
        # self.net_ir.load_state_dict(torch.load(encoder_ir_path))
        # self.net_vi.load_state_dict(torch.load(encoder_vi_path))
        # self.net_attention.load_state_dict(torch.load(attention_path))
        # self.net_feature_decoder.load_state_dict(torch.load(decoder_path))

    def forward(self, ir, vi):
        
        ir_features = self.net_ir(ir)
        vi_features = self.net_vi(vi)
        features = self.net_attention(ir_features, vi_features)
        fusion_images = self.net_feature_decoder(features)
        return fusion_images
