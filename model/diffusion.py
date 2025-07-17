from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from .module_util import default
from utils.sr_utils import SSIM, PerceptualLoss
from utils.hparams import hparams
import cv2

from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from .module_util import default
from utils.sr_utils import SSIM, PerceptualLoss
from utils.hparams import hparams
from utils.utils import random_affine_tensors, random_affine_tensors_with_inverse


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

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(num_diffusion_timesteps, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, encode_fusion, timesteps=1000, loss_type='l1'):
        super().__init__()
        self.denoise_fn = denoise_fn
        # condition net
        self.encode_fusion = encode_fusion
        self.ssim_loss = SSIM(window_size=11)
        if hparams['aux_percep_loss']:
            self.percep_loss_fn = [PerceptualLoss()]

        if hparams['beta_schedule'] == 'cosine':
            betas = cosine_beta_schedule(timesteps, s=hparams['beta_s'])
        if hparams['beta_schedule'] == 'linear':
            betas = get_beta_schedule(timesteps, beta_end=hparams['beta_end'])
            if hparams['res']:
                betas[-1] = 0.999

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.sample_tqdm = True

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, noise_pred, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def forward(self, img_ir, img_vi, mask_img, t=None, *args, **kwargs):
        # self.encode_fusion.eval()
        # with torch.no_grad():
        #     x = self.encode_fusion(img_ir, img_vi, True)

        img_vi_gray = RGB2Y(img_vi[:, 0, :, :].unsqueeze(1), img_vi[:, 1, :, :].unsqueeze(1),
                            img_vi[:, 2, :, :].unsqueeze(1))

        # x = encode_fusion_out.float()
        # x = torch.cat([img_vi_gray, img_ir], dim=1)
        x = img_ir
        # 仅仅配准的流程
        # en

        # with torch.no_grad():
        cond = self.encode_fusion(torch.cat([img_vi_gray, img_ir], dim=1))

        # visible_YCbCr = RGB2YCbCr(img_vi[:, 0, :, :].unsqueeze(1), img_vi[:, 1, :, :].unsqueeze(1),
        #                           img_vi[:, 2, :, :].unsqueeze(1))

        # visible_YCbCr = rgb_to_ycbcr(vi_image)
        # fusion_color = ycbcr_to_rgb(visible_YCbCr)

        # cond = YCbCr2RGB(cond, visible_YCbCr[:, 1, :, :].unsqueeze(1),
        #                          visible_YCbCr[:, 2, :, :].unsqueeze(1))
        # print(torch.max(cond))

        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() \
            if t is None else torch.LongTensor([t]).repeat(b).to(device)
        # print(t)



        # cond={}
        # cond['ir'] = rotate_ir
        # cond['vis'] = img_vi


        # 计算损失的cond的输入应为旋转后的img_ir


        p_losses_ir, x_tp1, noise_pred, x_t, x_t_gt, x_0 = self.p_losses(x, t, cond, *args, **kwargs)


        x = img_vi_gray

        # print(t)

        p_losses_vi, x_tp1, noise_pred, x_t, x_t_gt, x_0 = self.p_losses(x, t, cond, *args, **kwargs)

        total_loss, ir_pixel_loss_scale0, vi_pixel_loss_scale0, ir_grad_loss_scale0, vi_grad_loss_scale0, color_term = self.fusion_loss(img_ir, img_vi,cond,mask_img,1-mask_img)


        # x = img_vi_gray
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long() \
        #     if t is None else torch.LongTensor([t]).repeat(b).to(device)
        all_loss = 5 * (p_losses_ir + p_losses_vi) + total_loss
        ret = {'q': all_loss, 'q_ir':p_losses_ir,'q_vi':p_losses_vi,'fusion_loss':total_loss}
        # if not hparams['fix_encode']:
        #     if hparams['aux_l1_loss']:
        #         ret['aux_l1'] = F.l1_loss(encode_fusion_out, img_ir)+ F.l1_loss(encode_fusion_out, img_vi)
        #     if hparams['aux_ssim_loss']:
        #         ret['aux_ssim'] = 1 - self.ssim_loss(encode_fusion_out, img_ir)+1 - self.ssim_loss(encode_fusion_out, img_vi)
        #     if hparams['aux_percep_loss']:
        #         ret['aux_percep'] = self.percep_loss_fn[0](encode_fusion_out, img_ir)+self.percep_loss_fn[0](encode_fusion_out, img_vi)
        # x_recon = self.res2img(x_recon, img_lr_up)
        # x_tp1 = self.res2img(x_tp1, img_lr_up)
        # x_t = self.res2img(x_t, img_lr_up)
        # x_t_gt = self.res2img(x_t_gt, img_lr_up)
        return ret, (x_tp1, x_t_gt, x_t), t

    def p_losses(self, x_start, t, cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_tp1_gt = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_t_gt = self.q_sample(x_start=x_start, t=t - 1, noise=noise)
        noise_pred = self.denoise_fn(x_tp1_gt, t, cond)
        x_t_pred, x0_pred = self.p_sample(x_tp1_gt, t, cond, noise_pred=noise_pred)

        # 保存图片查看中间生成的结果
        # print(x0_pred)
        # x0_pred_out = np.round(x0_pred[0].cpu().numpy().transpose(1, 2, 0).squeeze())
        # x0_pred_out.clip(min=0, max=255).astype(np.uint8)
        # cv2.imwrite(r"x0_pred.jpg", x0_pred_out)
        #
        # x_start_out = np.round(x_start[0].cpu().numpy().transpose(1, 2, 0).squeeze())
        # x_start_out.clip(min=0, max=255).astype(np.uint8)
        # cv2.imwrite(r"x0_start.jpg", x_start_out)

        # x0_pred_out = tensor2img(x0_pred)

        if self.loss_type == 'l1':
            loss = (noise - noise_pred).abs().mean()
            # loss = loss + (1-self.ssim_loss(x0_pred, x_start))*0.2+0.2*F.mse_loss(x0_pred, x_start)
            # print("loss")
            # print(1)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)
        elif self.loss_type == 'ssim':
            # loss = (noise - noise_pred).abs().mean()
            loss = 1 - self.ssim_loss(noise, noise_pred)
        else:
            raise NotImplementedError()
        return loss, x_tp1_gt, noise_pred, x_t_pred, x_t_gt, x0_pred

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        t_cond = (t[:, None, None, None] >= 0).float()
        t = t.clamp_min(0)
        return (
                       extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                       extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
               ) * t_cond + x_start * (1 - t_cond)

    @torch.no_grad()
    def p_sample(self, x, t, cond, noise_pred=None, clip_denoised=True, repeat_noise=False):
        # print(x.shape)
        # print(t.shape)
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
            x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0_pred

    @torch.no_grad()
    def sample(self, img_ir, img_vi, shape, save_intermediate=False):
        device = self.betas.device
        x = torch.cat([img_vi, img_ir], dim=1)
        b = shape[0]
        # shape[1]=2
        img = torch.randn(shape, device=device)

        # encode_fusion_out, cond = self.encode_fusion(img_ir,img_vi, True)
        # encode_fusion_out = img_ir.float()
        #
        # cond={}
        # cond['ir'] = img_ir
        # cond['vis'] = img_vi


        # 仅仅配准的流程
        # en
        # cond = self.encode_fusion(x)
        # img_vi_gray = RGB2Y(img_vi[:, 0, :, :].unsqueeze(1), img_vi[:, 1, :, :].unsqueeze(1),
        #                     img_vi[:, 2, :, :].unsqueeze(1))
        cond = self.encode_fusion(torch.cat([img_vi, img_ir], dim=1))

        # visible_YCbCr = RGB2YCbCr(img_vi[:, 0, :, :].unsqueeze(1), img_vi[:, 1, :, :].unsqueeze(1),
        #                           img_vi[:, 2, :, :].unsqueeze(1))

        # visible_YCbCr = rgb_to_ycbcr(vi_image)
        # fusion_color = ycbcr_to_rgb(visible_YCbCr)

        # cond = YCbCr2RGB(cond, visible_YCbCr[:, 1, :, :].unsqueeze(1),
        #                  visible_YCbCr[:, 2, :, :].unsqueeze(1))



        it = reversed(range(0, self.num_timesteps))
        if self.sample_tqdm:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)
        images = []
        for i in it:
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond)
            if save_intermediate:
                img_ = img
                x_recon_ = x_recon
                images.append((img_.cpu(), x_recon_.cpu()))
        img = img
        if save_intermediate:
            return img, cond, images
        else:
            return img, cond

    @torch.no_grad()
    def interpolate(self, x1, x2, img_ir, img_vi, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        encode_fusion_out, cond = self.encode_fusion(img_ir, img_vi, True)


        assert x1.shape == x2.shape

        # x1 = self.img2res(x1, img_lr_up)
        # x2 = self.img2res(x2, img_lr_up)

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond)

        # img = img
        return img

    # def res2img(self, img_, img_lr_up, clip_input=None):
    #     if clip_input is None:
    #         clip_input = hparams['clip_input']
    #     if hparams['res']:
    #         if clip_input:
    #             img_ = img_.clamp(-1, 1)
    #         img_ = img_ / hparams['res_rescale'] + img_lr_up
    #     return img_
    #
    # def img2res(self, x, img_lr_up, clip_input=None):
    #     if clip_input is None:
    #         clip_input = hparams['clip_input']
    #     if hparams['res']:
    #         x = (x - img_lr_up) * hparams['res_rescale']
    #         if clip_input:
    #             x = x.clamp(-1, 1)
    #     return x

def tensor2img(img):
    # img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
    img = np.round((img.permute(0, 2, 3, 1).cpu().numpy()) * 255)
    img = img.clip(min=0, max=255).astype(np.uint8)
    return img