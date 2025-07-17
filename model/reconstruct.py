# 参考SDNet 构造重构的网络
import torch
import torch.nn as nn

class SDNet_decoder(nn.Module):
    def __init__(self):
        super(SDNet_decoder,self).__init__()

        self.decom = nn.Sequential((nn.Conv2d(1, 128, 1, 1, 0)), nn.LeakyReLU())
        self.conv51 = nn.Sequential((nn.Conv2d(128, 16, 3, 1, 1)), nn.LeakyReLU())
        self.conv52 = nn.Sequential((nn.Conv2d(128, 16, 3, 1, 1)), nn.LeakyReLU())

        self.conv61 = nn.Sequential((nn.Conv2d(16, 4, 3, 1, 1)), nn.LeakyReLU())
        self.conv62 = nn.Sequential((nn.Conv2d(16, 4, 3, 1, 1)), nn.LeakyReLU())

        self.conv71 = nn.Sequential((nn.Conv2d(4, 1, 3, 1, 1)), nn.Tanh())
        self.conv72 = nn.Sequential((nn.Conv2d(4, 1, 3, 1, 1)), nn.Tanh())
    
    def forward(self, x_fuse):
        x1_de = self.conv71(self.conv61(self.conv51(self.decom(x_fuse))))
        x2_de = self.conv72(self.conv62(self.conv52(self.decom(x_fuse))))
        return x1_de, x2_de