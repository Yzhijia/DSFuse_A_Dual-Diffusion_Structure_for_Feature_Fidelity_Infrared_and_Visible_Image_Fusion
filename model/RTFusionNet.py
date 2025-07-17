import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.transformer_base import RSTB,PatchEmbed,PatchUnEmbed,Upsample, UpsampleOneStep
from thop import profile

class RTFusionNet(nn.Module):

    def __init__(self,img_size=64, patch_size=1, in_chans=2, out_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, img_range=1., resi_connection='1conv',
                 **kwargs):
        super(RTFusionNet,self).__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 4:
            rgb_mean = (1, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 4, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)+0.5
            print(self.mean)
        print("****************************using RTFusionNet as backbone*******************************")
        self.window_size = window_size

        # shollow feature extraction 1 conv
        self.conv_first = nn.Sequential(nn.Conv2d(num_in_ch,embed_dim,3,1,1),
                                        nn.BatchNorm2d(embed_dim),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True))


        # deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
                                      embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge overlapping patches into image
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
                                          embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
            # layer = CRSTB(dim=embed_dim,
            #               input_resolution=(patches_resolution[0],
            #                                 patches_resolution[1]),
            #               depth=depths[i_layer],
            #               num_heads=num_heads[i_layer],
            #               window_size=window_size,
            #               mlp_ratio=self.mlp_ratio,
            #               qkv_bias=qkv_bias, qk_scale=qk_scale,
            #               drop=drop_rate, attn_drop=attn_drop_rate,
            #               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            #               # no impact on SR results
            #               norm_layer=norm_layer,
            #               downsample=None,
            #               use_checkpoint=use_checkpoint,
            #               img_size=img_size,
            #               patch_size=patch_size,
            #               resi_connection=resi_connection
            #               )
            # self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim,embed_dim,3,1,1),
                                        nn.BatchNorm2d(embed_dim),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.BatchNorm2d(embed_dim//4),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.BatchNorm2d(embed_dim//4),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
                                                 nn.BatchNorm2d(embed_dim),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))

        ### 源代码中 high quality image reconstruction 部分删除与否存疑

        self.conv_last = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                       nn.BatchNorm2d(num_feat),
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                       nn.Conv2d(num_feat, num_out_ch, 3, 1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_last(x)

        # x = x / self.img_range + self.mean
        # x = torch.tanh(x)
        x = (torch.tanh(x)+1)/2
        return x[ :, :, :H, :W]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        # flops += self.upsample.flops()
        return flops

if __name__ == '__main__':
    RTFsetting = {}
    RTFsetting['upscale'] = 1
    RTFsetting['in_chans'] = 1
    RTFsetting['img_size'] = 128
    RTFsetting['window_size'] = 8
    RTFsetting['img_range'] = 1.0
    RTFsetting['embed_dim'] = 60
    RTFsetting['num_heads'] = [6, 6, 6, 6]
    RTFsetting['mlp_ratio'] = 2
    RTFsetting['upsampler'] = None
    RTFsetting['resi_connection'] = "1conv"

    model = RTFusionNet(in_chans=2,
                        img_size=RTFsetting['img_size'],
                        window_size=RTFsetting['window_size'],
                        img_range=RTFsetting['img_range'],
                        embed_dim=RTFsetting['embed_dim'],
                        depths=[6,6],
                        num_heads=RTFsetting['num_heads'],
                        mlp_ratio=RTFsetting['mlp_ratio'],
                        resi_connection=RTFsetting['resi_connection'])
    dummy_input = torch.randn(1, 2, 512, 512)
    flops, params = profile(model, (dummy_input,))
    print('flops: %.6f M, params: %.6f M' % (flops / 1000000.0, params / 1000000.0))


