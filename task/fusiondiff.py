import os.path
import sys
sys.path.append(r"/home/yzj/code/transfusion_diffusion/")
import torch
from model.net import Unet, RRDBNet, encode_fusion
from model.diffusion2 import GaussianDiffusion as GaussianDiffusion2
from model.diffusion_swin import GaussianDiffusion as GaussianDiffusion_swin
from model.diffusion import GaussianDiffusion as GaussianDiffusion1
from task.trainer import Trainer, opt
from task.trainer_split import Trainer as Trainer_split
from task.trainer_swin import Trainer as Trainer_swin
from task.trainer_split import opt as opt_split
from utils.hparams import hparams
from model.SwinFusion import SwinFusion
from utils.utils import load_ckpt
from utils.utils import random_affine_tensors
import torch.nn.functional as F
from model.diffusion_split import GaussianDiffusion as GaussianDiffusion_split
from model.swinfusion_loss import loss_SwinFusion, loss_SwinFusion_split
from model.RTFusionNet import RTFusionNet
from model.ItFusionNet import ItFusionNet


# measure: 类似于损失函数， 可能需要更改
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

#
class FusionDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        # print(dim_mults)
        dim_mults = [int(x) for x in dim_mults.split('|')]
        denoise_fn_ir = Unet(
            hidden_size, out_dim=1, cond_dim=1, dim_mults=dim_mults)
        denoise_fn_vi = Unet(
            hidden_size, out_dim=1, cond_dim=1, dim_mults=dim_mults)
        # rrdb = RRDBNet(4, 3, hparams['fusion_num_feat'], hparams['fusion_num_block'],
        #                hparams['fusion_num_feat'] // 2)
        rrdb = RRDBNet(2, 1, 32, 3, 16)
        # rrdb = self.define_model_swin()
        # rrdb.load_state_dict(torch.load(
        #     r"epoch61_Fusion_decoder.pth",
        #     map_location='cuda:0'))
        # rrdb = torch.load(r"epoch61_Fusion_decoder.pth", map_location='cuda:0')
        # rrdb.load_state_dict(torch.load(r"cond.pth"))
        # fusion = encode_fusion(dim=hparams['fusion_num_feat'], depth=hparams['fusion_num_block'])
        # if hparams['fusion_ckpt'] != '' and os.path.exists(hparams['fusion_ckpt']):
        #     load_ckpt(fusion, hparams['fusion_ckpt'])
        #
        # # 如果固定融合的编码器，预载入参数
        # if hparams['fix_encode']:
        #     ck_pt_fusion = torch.load(hparams["fusion_weights_path"])
        #     print(ck_pt_fusion)
        #     fusion.load_state_dict(ck_pt_fusion if not hparams['fusion_from_checkpoint'] else ck_pt_fusion['g'])
        #



        self.model = GaussianDiffusion2(
            denoise_fn_ir=denoise_fn_ir,
            denoise_fn_vi=denoise_fn_vi,
            encode_fusion=rrdb,
            timesteps=hparams['timesteps'],
            loss_type=hparams['loss_type']
        )
        self.global_step = 0
        return self.model

    def define_model_swin(self):
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

    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0
        img_ir = sample['img_ir']
        # rotate_ir = random_affine_tensors(img_ir, img_ir.shape[0])
        img_vi = sample['img_vi']
        img_vi_gray = RGB2Y(img_vi[:, 0, :, :].unsqueeze(1), img_vi[:, 1, :, :].unsqueeze(1),
                            img_vi[:, 2, :, :].unsqueeze(1))
        # grid, fusion_out = self.model.sample(rotate_ir, img_vi, torch.cat([img_ir,img_ir],dim=1).shape)
        # img_sr = F.grid_sample(rotate_ir.float(), grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
        # print(rotate_ir.shape)
        # print(img_vi.shape)
        # img_sr, fusion_out = self.model.sample(img_ir, img_vi_gray, torch.cat([img_vi_gray, img_ir], dim=1).shape)
        img_sr, fusion_out = self.model.sample(img_ir, img_vi_gray, img_ir.shape)
        for b in range(img_sr[0].shape[0]):
            # print("fusion_out:"+str(fusion_out.shape))
            # print("img_ir:" + str(img_ir.shape))
            # print("img_vi:" + str(img_vi.shape))
            # s = self.measure.measure(fusion_out[b], torch.cat((img_ir[b],img_ir[b],img_ir[b]),dim=1), img_vi[b])
            # ret['psnr'] += s['psnr']
            # ret['ssim'] += s['ssim']
            # ret['lpips'] += s['lpips']
            # ret['lr_psnr'] += s['lr_psnr']
            # ret['n_samples'] += 1

            ret['psnr'] += 0
            ret['ssim'] += 0
            ret['lpips'] += 0
            ret['lr_psnr'] += 0
            ret['n_samples'] += 1

        return img_sr, fusion_out, ret

    def build_optimizer(self, model):
        params = list(model.named_parameters())
        if not hparams['fix_encode']:
            params = [p for p in params if 'encode_fusion' not in p[0]]
        params = [p[1] for p in params]
        return torch.optim.Adam(params, lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def training_step(self, batch):
        img_ir = batch['img_ir']
        img_vi = batch['img_vi']
        mask_img = batch["mask_img"]
        losses, _, _ = self.model(img_ir, img_vi, mask_img)
        total_loss = sum(losses.values())
        return losses, total_loss



class FusionDiffTrainer_Swin(Trainer_swin):
    def build_model(self):
        print(hparams)
        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        # print(dim_mults)
        dim_mults = [int(x) for x in dim_mults.split('|')]
        denoise_fn_ir = Unet(
            hidden_size, out_dim=1, cond_dim=1, dim_mults=dim_mults)
        denoise_fn_vi = Unet(
            hidden_size, out_dim=1, cond_dim=1, dim_mults=dim_mults)
        # rrdb = RRDBNet(4, 3, hparams['fusion_num_feat'], hparams['fusion_num_block'],
        #                hparams['fusion_num_feat'] // 2)
        # rrdb = RRDBNet(2, 1, 32, 3, 16)
        rrdb = self.define_model_fusion()
        # rrdb.load_state_dict(torch.load(
        #     r"epoch61_Fusion_decoder.pth",
        #     map_location='cuda:0'))
        # rrdb = torch.load(r"epoch61_Fusion_decoder.pth", map_location='cuda:0')
        # rrdb.load_state_dict(torch.load(r"cond.pth"))
        # fusion = encode_fusion(dim=hparams['fusion_num_feat'], depth=hparams['fusion_num_block'])
        # if hparams['fusion_ckpt'] != '' and os.path.exists(hparams['fusion_ckpt']):
        #     load_ckpt(fusion, hparams['fusion_ckpt'])
        #
        # # 如果固定融合的编码器，预载入参数
        # if hparams['fix_encode']:
        #     ck_pt_fusion = torch.load(hparams["fusion_weights_path"])
        #     print(ck_pt_fusion)
        #     fusion.load_state_dict(ck_pt_fusion if not hparams['fusion_from_checkpoint'] else ck_pt_fusion['g'])
        #



        self.model = GaussianDiffusion_swin(
            denoise_fn_ir=denoise_fn_ir,
            denoise_fn_vi=denoise_fn_vi,
            encode_fusion=rrdb,
            timesteps=hparams['timesteps'],
            loss_type=hparams['loss_type']
        )
        self.global_step = 0
        return self.model

    def define_model_swin(self):
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

    def define_model_RTF(self):
        model = RTFusionNet(in_chans=2,
                            img_size=opt['img_size'],
                            window_size=opt['window_size'],
                            img_range=opt['img_range'],
                            embed_dim=opt['embed_dim'],
                            num_heads=opt['num_heads'],
                            mlp_ratio=opt['mlp_ratio'],
                            resi_connection=opt['resi_connection'])
        print("############### Build the model using RTF Model#############")
        return model
    
    def define_model_fusion(self):
        if hparams["fusion_model_type"] == "SwinFusion":
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
        elif hparams["fusion_model_type"] == "RTFusionNet":
            model = RTFusionNet(in_chans=2,
                            img_size=opt['img_size'],
                            window_size=opt['window_size'],
                            img_range=opt['img_range'],
                            embed_dim=opt['embed_dim'],
                            num_heads=opt['num_heads'],
                            mlp_ratio=opt['mlp_ratio'],
                            resi_connection=opt['resi_connection'])
            return model
        elif hparams["fusion_model_type"] == "ItFusionNet":
            model = ItFusionNet()
            return model

    

    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0
        img_ir = sample['img_ir']
        # rotate_ir = random_affine_tensors(img_ir, img_ir.shape[0])
        img_vi = sample['img_vi']
        img_vi_gray = RGB2Y(img_vi[:, 0, :, :].unsqueeze(1), img_vi[:, 1, :, :].unsqueeze(1),
                            img_vi[:, 2, :, :].unsqueeze(1))
        # grid, fusion_out = self.model.sample(rotate_ir, img_vi, torch.cat([img_ir,img_ir],dim=1).shape)
        # img_sr = F.grid_sample(rotate_ir.float(), grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
        # print(rotate_ir.shape)
        # print(img_vi.shape)
        # img_sr, fusion_out = self.model.sample(img_ir, img_vi_gray, torch.cat([img_vi_gray, img_ir], dim=1).shape)
        img_sr, fusion_out = self.model.sample(img_ir, img_vi_gray, img_ir.shape)
        for b in range(img_sr[0].shape[0]):
            # print("fusion_out:"+str(fusion_out.shape))
            # print("img_ir:" + str(img_ir.shape))
            # print("img_vi:" + str(img_vi.shape))
            # s = self.measure.measure(fusion_out[b], torch.cat((img_ir[b],img_ir[b],img_ir[b]),dim=1), img_vi[b])
            # ret['psnr'] += s['psnr']
            # ret['ssim'] += s['ssim']
            # ret['lpips'] += s['lpips']
            # ret['lr_psnr'] += s['lr_psnr']
            # ret['n_samples'] += 1

            ret['psnr'] += 0
            ret['ssim'] += 0
            ret['lpips'] += 0
            ret['lr_psnr'] += 0
            ret['n_samples'] += 1

        return img_sr, fusion_out, ret

    def build_optimizer(self, model):
        params = list(model.named_parameters())
        if not hparams['fix_encode']:
            params = [p for p in params if 'encode_fusion' not in p[0]]
        params = [p[1] for p in params]
        return torch.optim.Adam(params, lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def training_step(self, batch):
        img_ir = batch['img_ir']
        img_vi = batch['img_vi']
        mask_img = batch["mask_img"]
        losses, _, _ = self.model(img_ir, img_vi, mask_img)
        # total_loss = sum(losses.values())
        total_loss = losses['q']
        return losses, total_loss








class FusionDiffTrainer_Split(Trainer_split):
    def build_model(self):
        print(hparams)
        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        # print(dim_mults)
        dim_mults = [int(x) for x in dim_mults.split('|')]
        denoise_fn_ir = Unet(
            hidden_size, out_dim=1, cond_dim=1, dim_mults=dim_mults)
        denoise_fn_vi = Unet(
            hidden_size, out_dim=1, cond_dim=1, dim_mults=dim_mults)
        # rrdb = RRDBNet(4, 3, hparams['fusion_num_feat'], hparams['fusion_num_block'],
        #                hparams['fusion_num_feat'] // 2)
        # rrdb = RRDBNet(2, 1, 32, 3, 16)
        # rrdb = self.define_model_swin()
        # rrdb.load_state_dict(torch.load(
        #     r"epoch61_Fusion_decoder.pth",
        #     map_location='cuda:0'))
        # rrdb = torch.load(r"epoch61_Fusion_decoder.pth", map_location='cuda:0')
        # rrdb.load_state_dict(torch.load(r"cond.pth"))
        # fusion = encode_fusion(dim=hparams['fusion_num_feat'], depth=hparams['fusion_num_block'])
        # if hparams['fusion_ckpt'] != '' and os.path.exists(hparams['fusion_ckpt']):
        #     load_ckpt(fusion, hparams['fusion_ckpt'])
        #
        # # 如果固定融合的编码器，预载入参数
        # if hparams['fix_encode']:
        #     ck_pt_fusion = torch.load(hparams["fusion_weights_path"])
        #     print(ck_pt_fusion)
        #     fusion.load_state_dict(ck_pt_fusion if not hparams['fusion_from_checkpoint'] else ck_pt_fusion['g'])
        #



        self.model_ir = GaussianDiffusion_split(
            denoise_fn=denoise_fn_ir,
            timesteps=hparams['timesteps'],
            loss_type=hparams['loss_type']
        )

        self.model_vi = GaussianDiffusion_split(
            denoise_fn=denoise_fn_vi,
            timesteps=hparams['timesteps'],
            loss_type=hparams['loss_type']
        )

        # self.model_fusion = self.define_model_swin()

        self.model_fusion = self.define_model_RTF()

        self.fusion_loss = loss_SwinFusion_split()


        self.global_step = 0
        return self.model_ir, self.model_vi, self.model_fusion

    def define_model_swin(self):
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

    def define_model_RTF(self):
        model = RTFusionNet(in_chans=2,
                            img_size=opt['img_size'],
                            window_size=opt['window_size'],
                            img_range=opt['img_range'],
                            embed_dim=opt['embed_dim'],
                            num_heads=opt['num_heads'],
                            mlp_ratio=opt['mlp_ratio'],
                            resi_connection=opt['resi_connection'])
        return model


    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0
        img_ir = sample['img_ir']
        # rotate_ir = random_affine_tensors(img_ir, img_ir.shape[0])
        img_vi = sample['img_vi']
        img_vi_gray = RGB2Y(img_vi[:, 0, :, :].unsqueeze(1), img_vi[:, 1, :, :].unsqueeze(1),
                            img_vi[:, 2, :, :].unsqueeze(1))
        x = torch.cat([img_vi_gray, img_ir], dim=1)
        fusion = self.model_fusion(x)

        # grid, fusion_out = self.model.sample(rotate_ir, img_vi, torch.cat([img_ir,img_ir],dim=1).shape)
        # img_sr = F.grid_sample(rotate_ir.float(), grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
        # print(rotate_ir.shape)
        # print(img_vi.shape)
        # img_sr, fusion_out = self.model.sample(img_ir, img_vi_gray, torch.cat([img_vi_gray, img_ir], dim=1).shape)
        img_sr_ir, fusion_out = self.model_ir.sample(fusion, fusion.shape)
        img_sr_vi, fusion_out = self.model_vi.sample(fusion, fusion.shape)
        for b in range(img_sr_ir.shape[0]):
            # print("fusion_out:"+str(fusion_out.shape))
            # print("img_ir:" + str(img_ir.shape))
            # print("img_vi:" + str(img_vi.shape))
            # s = self.measure.measure(fusion_out[b], torch.cat((img_ir[b],img_ir[b],img_ir[b]),dim=1), img_vi[b])
            # ret['psnr'] += s['psnr']
            # ret['ssim'] += s['ssim']
            # ret['lpips'] += s['lpips']
            # ret['lr_psnr'] += s['lr_psnr']
            # ret['n_samples'] += 1

            ret['psnr'] += 0
            ret['ssim'] += 0
            ret['lpips'] += 0
            ret['lr_psnr'] += 0
            ret['n_samples'] += 1

        return (img_sr_ir, img_sr_vi), fusion_out, ret

    def build_optimizer(self, model):
        params = list(model.named_parameters())
        if not hparams['fix_encode']:
            params = [p for p in params if 'encode_fusion' not in p[0]]
        params = [p[1] for p in params]
        return torch.optim.Adam(params, lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def training_step_diffusion(self, batch):
        img_ir = batch['img_ir']
        img_vi = batch['img_vi']
        mask_img = batch["mask_img"]

        img_vi_gray = RGB2Y(img_vi[:, 0, :, :].unsqueeze(1), img_vi[:, 1, :, :].unsqueeze(1),
                            img_vi[:, 2, :, :].unsqueeze(1))

        with torch.no_grad():
            x = torch.cat([img_vi_gray, img_ir], dim=1)
            fusion = self.model_fusion(x)


        losses_ir, _, _ = self.model_ir(img_ir, fusion)
        losses_vi, _, _ = self.model_vi(img_vi_gray, fusion)
        total_loss_ir = sum(losses_ir.values())
        total_loss_vi = sum(losses_vi.values())
        total_loss = total_loss_vi + total_loss_ir

        return {'q_ir':total_loss_ir, 'q_vi':total_loss_vi}, total_loss

    def training_step_fusion(self, batch):
        img_ir = batch['img_ir']
        img_vi = batch['img_vi']
        mask_img = batch["mask_img"]

        img_vi_gray = RGB2Y(img_vi[:, 0, :, :].unsqueeze(1), img_vi[:, 1, :, :].unsqueeze(1),
                            img_vi[:, 2, :, :].unsqueeze(1))


        x = torch.cat([img_vi_gray, img_ir], dim=1)
        fusion_image = self.model_fusion(x)

        with torch.no_grad():
            loss_ir, (_,_,img_ir_r), _ = self.model_ir(img_ir, fusion_image)
            loss_vi, (_,_,img_vi_r), _ = self.model_vi(img_vi_gray, fusion_image)

        total_loss, loss_gradient, loss_l1, loss_SSIM, loss_int = self.fusion_loss(img_ir, img_vi_gray, fusion_image, img_ir_r, img_vi_r)

        return total_loss,{'fusion_loss':total_loss, 'loss_gradient': loss_gradient, 'loss_l1': loss_l1, 'loss_SSIM':loss_SSIM, 'loss_int':loss_int}





