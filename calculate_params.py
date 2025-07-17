from torchstat import stat
from ptflops import get_model_complexity_info
from model.RTFusionNet import RTFusionNet
from model.SwinFusion import SwinFusion
import torch
from thop import profile
from thop import clever_format

opt = {}
opt['upscale'] = 1
opt['in_chans'] = 1
opt['img_size'] = 128
opt['window_size'] = 8
opt['img_range'] = 1.0
opt['embed_dim'] = 60
opt['num_heads'] = [6, 6, 6, 6]
opt['mlp_ratio'] = 2
opt['upsampler'] = None
opt['resi_connection'] = "1conv"


model_SwinFusion = SwinFusion(upscale=opt['upscale'],
                           in_chans=opt['in_chans'],
                           img_size=opt['img_size'],
                           window_size=opt['window_size'],
                           img_range=opt['img_range'],
                           #    depths=opt_net['depths'],
                           embed_dim=opt['embed_dim'],
                           num_heads=opt['num_heads'],
                           mlp_ratio=opt['mlp_ratio'],
                           upsampler=opt['upsampler'],
                           resi_connection=opt['resi_connection'])

model_RTFFusion = RTFusionNet(in_chans=2,
                            img_size=opt['img_size'],
                            window_size=opt['window_size'],
                            img_range=opt['img_range'],
                            embed_dim=opt['embed_dim'],
                            depths=[6,6],
                            num_heads=opt['num_heads'],
                            mlp_ratio=opt['mlp_ratio'],
                            resi_connection=opt['resi_connection'])
print("******************************state the SwinFusion*************************")

input1 = torch.randn(1,1,512,512)
input2 = torch.randn(1,1,512,512)
# stat(model_SwinFusion, [(1,1,224,224),(1,1,224,224)])
flops, params = profile(model_SwinFusion, inputs=(input1,input2))
print(flops, params)
flops, params = clever_format([flops, params],"%.3f")
print(flops, params)

print("******************************state the SwinFusion*************************")

input1 = torch.randn(1,2,224,224)
# input2 = torch.randn(1,1,224,224)
# stat(model_SwinFusion, [(1,1,224,224),(1,1,224,224)])
flops, params = profile(model_RTFFusion, inputs=(input1,))
print(flops, params)
flops, params = clever_format([flops, params],"%.3f")
print(flops, params)





# macs_SF, params_SF = get_model_complexity_info(model_SwinFusion, (1,224,224),as_strings=True,
#                                                print_per_layer_stat=True, verbose=True)
# macs_RT, params_RT = get_model_complexity_info(model_SwinFusion, (1,224,224),as_strings=True,
#                                                print_per_layer_stat=True, verbose=True)
#

