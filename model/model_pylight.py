import torch
import numpy as np
import pytorch_lightning as pl
from utils.hparams import hparams
from model.net import Unet, RRDBNet, encode_fusion
from task.trainer import opt
from model.diffusion_swin import GaussianDiffusion as GaussianDiffusion_swin
from model.RTFusionNet import RTFusionNet
from model.SwinFusion import SwinFusion
from model.ItFusionNet import ItFusionNet

class PL_diffusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        dim_mults = [int(x) for x in dim_mults.split('|')]
        
        denoise_fn_ir = Unet(
            hidden_size, out_dim=1, cond_dim=1, dim_mults=dim_mults)
        denoise_fn_vi = Unet(
            hidden_size, out_dim=1, cond_dim=1, dim_mults=dim_mults)

        rrdb = self.define_model_fusion()


        self.model = GaussianDiffusion_swin(
            denoise_fn_ir=denoise_fn_ir,
            denoise_fn_vi=denoise_fn_vi,
            encode_fusion=rrdb,
            timesteps=hparams['timesteps'],
            loss_type=hparams['loss_type']
        )       
        self.global_step = 0
    
    def define_model_fusion(self):
        if opt.fusion_model_type == "SwinFusion":
            model = SwinFusion(upscale=opt['upscale'],
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
            return model
        elif opt.fusion_model_type == "RTFusionNet":
            model = RTFusionNet(in_chans=2,
                            img_size=opt['img_size'],
                            window_size=opt['window_size'],
                            img_range=opt['img_range'],
                            embed_dim=opt['embed_dim'],
                            num_heads=opt['num_heads'],
                            mlp_ratio=opt['mlp_ratio'],
                            resi_connection=opt['resi_connection'])
            return model
        elif opt.fusion_model_type == "ItFusionNet":
            model = ItFusionNet()
            return model
    
    def build_optimizer(self, model):
        params = list(model.named_parameters())
        if not hparams['fix_encode']:
            params = [p for p in params if 'encode_fusion' not in p[0]]
        params = [p[1] for p in params]
        return torch.optim.Adam(params, lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)


    def configure_optimizers(self):
        optimizer = self.build_optimizer(self.model)
        scheduler = self.build_scheduler(optimizer)
        return [optimizer],[scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        img_ir = batch['img_ir']
        img_vi = batch['img_vi']
        mask_img = batch["mask_img"]
        losses, _, _ = self.model(img_ir, img_vi, mask_img)
        total_loss = losses['q']

        # 记录和显示损失

        return {'loss': total_loss}


    # def 

    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     if self.trainer

    







