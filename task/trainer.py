import importlib
import os
import subprocess
import sys
sys.path.append(r"/home/yzj/code/transfusion_diffusion/")
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.hparams import hparams, set_hparams, opt
import numpy as np
from dataset.dataset_make import Fusion_Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.utils import plot_img, move_to_cuda, load_checkpoint, save_checkpoint, tensors_to_scalars, load_ckpt, Measure
import task.fusiondiff
from utils.utils_dist import init_dist, get_dist_info
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.utils import random_affine_tensors, random_affine_tensors_with_inverse
from model.net import Unet, RRDBNet, encode_fusion
import torch.nn.functional as F
from model.trans import tensor2img
import cv2



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



class Trainer:
    def __init__(self):
        self.logger = self.build_tensorboard(save_dir=hparams['work_dir'], name='tb_logs')
        self.measure = Measure()
        self.dataset_cls = None
        self.metric_keys = ['psnr', 'ssim', 'lpips', 'lr_psnr']
        self.work_dir = hparams['work_dir']
        self.first_val = True

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir, **kwargs)

    def build_train_dataloader(self):
        dataset = Fusion_Dataset(hparams['path_ir'], hparams['path_vi'], hparams['path_mask'],
                                 hparams['crop_size'], hparams['upscale_factor'], prefix='train_mode2')
        if opt['rank'] == 0:
            print('Number of train images: {:,d}'.format(len(dataset)))
        train_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        train_loader = torch.utils.data.DataLoader(dataset,
                                  batch_size=opt['batch_size'] // opt['num_gpu'],
                                  shuffle=False,
                                  num_workers=opt['num_workers'] // opt['num_gpu'],
                                  drop_last=True,
                                  pin_memory=True,
                                  sampler=train_sampler)

        return train_loader
        # dataset = self.dataset_cls('train')
        # return torch.utils.data.DataLoader(
        #     dataset, batch_size=hparams['batch_size'], shuffle=True,
        #     pin_memory=False, num_workers=hparams['num_workers'])

    def build_val_dataloader(self):
        dataset = Fusion_Dataset(hparams['path_val_ir'], hparams['path_val_vi'], hparams['path_val_mask'],
                                 hparams['crop_size'], hparams['upscale_factor'], prefix='valid')
        return torch.utils.data.DataLoader(
            dataset, batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)

    def build_test_dataloader(self):
        # dataset = Fusion_Dataset(hparams['path_test_ir'], hparams['path_test_vi'], hparams['path_test_mask'],
        #                          hparams['crop_size'], hparams['upscale_factor'], prefix='test')
        print("test_path:"+hparams['path_test_ir'])
        dataset = Fusion_Dataset(r"/data2/yzj/dataset/test_data/ir/", "/data2/yzj/dataset/test_data/vi/", r"/data2/yzj/dataset/test_data/ir/",
                                 hparams['crop_size'], hparams['upscale_factor'], prefix='test_full_scale')
        # dataset = Fusion_Dataset(r"/home/yzj/dataset/LLVIP/IR/", r"/home/yzj/dataset/LLVIP/VI/", r"/home/yzj/dataset/LLVIP/IR/",
        #                          hparams['crop_size'], hparams['upscale_factor'], prefix='test_full_scale')
        return torch.utils.data.DataLoader(
            dataset, batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)

    def build_model(self):
        raise NotImplementedError

    def sample_and_test(self, sample):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    def train(self):
        # 多卡训练设置
        # if mp.get_start_method(allow_none=True) is None:
        #     mp.set_start_method('spawn')
        #
        # rank = int(os.environ['RANK'])


        model = self.build_model()
        optimizer_d_ir = self.build_optimizer(model.denoise_fn_ir)
        optimizer_d_vi = self.build_optimizer(model.denoise_fn_vi)
        optimizer_f = self.build_optimizer(model.encode_fusion)
        self.global_step = training_step = load_checkpoint(model, optimizer_d_ir, optimizer_d_vi, optimizer_f, hparams['work_dir'])
        # load_checkpoint(model)
        print("traing_step:"+str(training_step))
        self.scheduler_d_ir = scheduler_d_ir = self.build_scheduler(optimizer_d_ir)
        self.scheduler_d_vi = scheduler_d_vi = self.build_scheduler(optimizer_d_vi)
        self.scheduler_f = scheduler_f = self.build_scheduler(optimizer_f)

        scheduler_d_ir.step(training_step)
        scheduler_d_vi.step(training_step)
        scheduler_f.step(training_step)
        dataloader = self.build_train_dataloader()
        device_id = torch.cuda.current_device()

        while self.global_step < hparams['max_updates']:
            train_pbar = tqdm(dataloader, initial=training_step, total=float('inf'),
                              dynamic_ncols=True, unit='step')
            for batch in train_pbar:
                # print(type(batch["img_ir"]))
                if opt["rank"] == 0:
                    if training_step % hparams['val_check_interval'] == 0:
                        with torch.no_grad():
                            model.eval()
                            self.validate(training_step)
                        # save_checkpoint(model.encode_fusion, optimizer, self.work_dir+"/fusion/", training_step, hparams['num_ckpt_keep'])
                        save_checkpoint(model, optimizer_d_ir,optimizer_d_vi,optimizer_f, self.work_dir, training_step, hparams['num_ckpt_keep'])
                model.train()
                batch = move_to_cuda(batch, device_id)

                losses, total_loss = self.training_step(batch)
                # print(losses)

                optimizer_d_ir.zero_grad()
                optimizer_f.zero_grad()
                optimizer_d_vi.zero_grad()
                # loss_ir = losses['q_ir']
                total_loss.backward()

                optimizer_d_ir.step()


                # loss_vi = losses['q_vi']

                optimizer_d_vi.step()


                # loss_fusion = losses['fusion_loss']

                optimizer_f.step()

                training_step += 1
                scheduler_d_ir.step(training_step)
                scheduler_d_vi.step(training_step)
                scheduler_f.step(training_step)
                self.global_step = training_step
                if opt["rank"] == 0:
                    if training_step % 100 == 0:
                        self.log_metrics({f'tr/{k}': v for k, v in losses.items()}, training_step)
                    train_pbar.set_postfix(**tensors_to_scalars(losses))
                # print("global_step:"+str(self.global_step))
                # print("go to next iteration!")
            # self.global_step = hparams['max_updates']

    def validate(self, training_step):
        device_id = torch.cuda.current_device()
        val_dataloader = self.build_val_dataloader()
        # print(hparams["batch_size"])
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for batch_idx, batch in pbar:
            if self.first_val and batch_idx > hparams['num_sanity_val_steps']:  # 每次运行的第一次validation只跑一小部分数据，来验证代码能否跑通
                break
            batch = move_to_cuda(batch, device_id)
            fusion, encode_out, ret = self.sample_and_test(batch)
            img_ir = batch['img_ir']
            img_vi = batch['img_vi']
            # img_lr_up = batch['img_lr_up']

            # if fusion is not None:
            #     self.logger.add_image(f'Pred_{batch_idx}', plot_img(fusion[0]), self.global_step)
            #     if hparams.get('aux_l1_loss'):
            #         self.logger.add_image(f'rrdb_out_{batch_idx}', plot_img(encode_out[0]), self.global_step)
            #     if self.global_step <= hparams['val_check_interval']:
            #         self.logger.add_image(f'HR_{batch_idx}', plot_img(img_ir[0]), self.global_step)
            #         self.logger.add_image(f'LR_{batch_idx}', plot_img(img_vi[0]), self.global_step)
                    # self.logger.add_image(f'BL_{batch_idx}', plot_img(img_lr_up[0]), self.global_step)
            metrics = {}
            metrics.update({k: np.mean(ret[k]) for k in self.metric_keys})
            pbar.set_postfix(**tensors_to_scalars(metrics))
        if hparams['infer']:
            print('Val results:', metrics)
        else:
            if not self.first_val:
                self.log_metrics({f'val/{k}': v for k, v in metrics.items()}, training_step)
                print('Val results:', metrics)
            else:
                print('Sanity val results:', metrics)
        self.first_val = False






    def test(self):
        model = self.build_model()
        optimizer_d_ir = self.build_optimizer(model.denoise_fn_ir)
        optimizer_d_vi = self.build_optimizer(model.denoise_fn_vi)
        optimizer_f = self.build_optimizer(model.encode_fusion)
        load_checkpoint(model, optimizer_d_ir,optimizer_d_vi,optimizer_f, hparams['work_dir'])
        optimizer = None
        device_id = torch.cuda.current_device()
        # 用最近的
        # self.global_step
        self.results = {k: 0 for k in self.metric_keys}
        self.n_samples = 0
        self.gen_dir = f"{hparams['work_dir']}/results_{self.global_step}_{hparams['gen_dir_name']}"
        if hparams['test_save_png']:
            subprocess.check_call(f'rm -rf {self.gen_dir}', shell=True)
            os.makedirs(f'{self.gen_dir}/outputs', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/SR', exist_ok=True)

        self.model.sample_tqdm = False
        torch.backends.cudnn.benchmark = False
        if hparams['test_save_png']:
            if hasattr(self.model.denoise_fn_ir, 'make_generation_fast_'):
                self.model.denoise_fn_ir.make_generation_fast_()
            os.makedirs(f'{self.gen_dir}/ENCODE', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/IR', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/VI', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/COMPARE', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/SR_VI', exist_ok=True)
            # os.makedirs(f'{self.gen_dir}/UP', exist_ok=True)

        with torch.no_grad():
            model.eval()
            test_dataloader = self.build_test_dataloader()
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for batch_idx, batch in pbar:
                move_to_cuda(batch, device_id)
                gen_dir = self.gen_dir
                item_names = batch['item_name']
                img_ir = batch['img_ir']
                # rotate_ir = img_ir
                # rotate_ir = random_affine_tensors(img_ir, img_ir.shape[0])
                # (grid, grid_inv)=random_affine_tensors_with_inverse(img_ir, img_ir.shape[0])
                # test_ir = F.grid_sample(img_ir, grid, mode='bilinear', padding_mode='zeros')
                # test_ir_inv = F.grid_sample(test_ir, grid_inv, mode='bilinear', padding_mode='zeros')
                # Image.fromarray(np.squeeze(tensor2img(test_ir))).save(
                #     f"/home/yzj/code/fusion_diffusion/checkpoints/1.png")
                # Image.fromarray(np.squeeze(tensor2img(test_ir_inv))).save(
                #     f"/home/yzj/code/fusion_diffusion/checkpoints/2.png")
                # print("图像保存成功！！！")


                # batch['img_ir'] = rotate_ir
                img_vi = batch['img_vi']
                # img_lr_up = batch['img_lr_up']
                # print("save_intermediate"+str(hparams['save_intermediate']))
                # if hparams['save_intermediate']:
                #     item_name = item_names[0]
                #     # img, encode_out, imgs = self.model.sample(
                #     #     img_ir, img_vi, img_ir.shape, save_intermediate=True)
                #
                #     # grid, encode_out, imgs = self.model.sample(
                #     #     rotate_ir, img_vi, torch.cat([img_ir,img_ir],dim=1).shape, save_intermediate=True)
                #
                #     # img = F.grid_sample(rotate_ir.float(), grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
                #
                #     img, encode_out, imgs = self.model.sample(
                #         img_ir, img_vi, torch.cat([img_vi, img_ir], dim=1).shape, save_intermediate=True)
                #
                #     os.makedirs(f"{gen_dir}/intermediate/{item_name}", exist_ok=True)
                #     # print('image_shape:'+str(self.tensor2img(img).shape))
                #     # print('gen_dir:'+gen_dir)
                #     Image.fromarray(np.squeeze(self.tensor2img(img))).save(f"{gen_dir}/intermediate/{item_name}/G.png")
                #
                #     for i, (m, x_recon) in enumerate(tqdm(imgs)):
                #         if i % (hparams['timesteps'] // 20) == 0 or i == hparams['timesteps'] - 1:
                #             t_batched = torch.stack([torch.tensor(i).to(img.device)] * img.shape[0])
                #             x_t = self.model.q_sample(encode_out[0], t=t_batched)
                #             Image.fromarray(np.squeeze(self.tensor2img(x_t))).save(
                #                 f"{gen_dir}/intermediate/{item_name}/noise1_{i:03d}.png")
                #             Image.fromarray(np.squeeze(self.tensor2img(m))).save(
                #                 f"{gen_dir}/intermediate/{item_name}/noise_{i:03d}.png")
                #             Image.fromarray(np.squeeze(self.tensor2img(x_recon))).save(
                #                 f"{gen_dir}/intermediate/{item_name}/{i:03d}.png")
                #     # return {}

                res = self.sample_and_test(batch)
                if len(res) == 3:
                    img_sr, encode_out, ret = res
                else:
                    img_sr, ret = res
                    encode_out = img_sr
                img_ir = batch['img_ir']
                # img_ir = rotate_ir
                img_vi = batch['img_vi']
                # img_lr_up = batch.get('img_lr_up', img_lr_up)
                if img_sr is not None:
                    img_sr_ir = img_sr[0]
                    img_sr_vi = img_sr[1]
                    metrics = list(self.metric_keys)
                    for k in metrics:
                        self.results[k] += ret[k]
                    self.n_samples += ret['n_samples']
                    # print({k: round(self.results[k] / self.n_samples, 3) for k in metrics}, 'total:', self.n_samples)
                    if hparams['test_save_png'] and img_sr_ir is not None:
                        # print(img_sr_ir.shape)
                        # img_sr = self.tensor2img(img_sr[:,1,:,:].unsqueeze(1))

                        # 赋予encode_out 颜色
                        visible_YCbCr = RGB2YCbCr(img_vi[:, 0, :, :].unsqueeze(1),
                                                  img_vi[:, 1, :, :].unsqueeze(1),
                                                  img_vi[:, 2, :, :].unsqueeze(1))

                        encode_color = YCbCr2RGB(encode_out, visible_YCbCr[:, 1, :, :].unsqueeze(1),
                                                 visible_YCbCr[:, 2, :, :].unsqueeze(1))

                        img_sr_ir = self.tensor2img(img_sr_ir)
                        img_sr_vi = self.tensor2img(img_sr_vi)
                        img_ir = self.tensor2img(img_ir)
                        img_vi = self.tensor2img(img_vi)
                        # print('shape of img_vi'+str(img_vi.shape))
                        # img_lr_up = self.tensor2img(img_lr_up)
                        encode_out = self.tensor2img(encode_out)
                        encode_color = self.tensor2img(encode_color)
                        # print(encode_out.shape)
                        # encode_out = cv2.cvtColor(np.squeeze(encode_out), cv2.COLOR_RGB2BGR)
                        for item_name, hr_p_ir, hr_p_vi, ir, vi, encode_o, encode_c in zip(
                                item_names, img_sr_ir, img_sr_vi, img_ir, img_vi, encode_out, encode_color):
                            # print('shape of img_vi' + str(vi.shape))
                            item_name = os.path.splitext(item_name)[0]

                            # print(vi.shape)
                            # compare = Image.fromarray(np.concatenate((hr_p, vi, vi), axis=2))
                            compare = Image.fromarray(np.squeeze(encode_c))
                            # hr_p = Image.fromarray(np.squeeze(hr_p)[:,:,0])
                            hr_p_ir = Image.fromarray(np.squeeze(hr_p_ir))
                            hr_p_vi = Image.fromarray(np.squeeze(hr_p_vi))
                            ir = Image.fromarray(np.squeeze(ir))
                            vi = Image.fromarray(np.squeeze(vi))


                            encode_o = Image.fromarray(np.squeeze(encode_o))
                            hr_p_ir.save(f"{gen_dir}/outputs/{item_name}[SR_IR].jpg")
                            hr_p_vi.save(f"{gen_dir}/outputs/{item_name}[SR_VI].jpg")
                            ir.save(f"{gen_dir}/outputs/{item_name}[IR].jpg")
                            vi.save(f"{gen_dir}/outputs/{item_name}[VI].jpg")
                            hr_p_ir.save(f"{gen_dir}/SR/{item_name}.jpg")
                            hr_p_vi.save(f"{gen_dir}/SR_VI/{item_name}.jpg")
                            ir.save(f"{gen_dir}/IR/{item_name}.jpg")
                            vi.save(f"{gen_dir}/VI/{item_name}.jpg")
                            encode_o.save(f"{gen_dir}/ENCODE/{item_name}.jpg")
                            compare.save(f"{gen_dir}/COMPARE/{item_name}.jpg")





    # utils
    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    @staticmethod
    def tensor2img(img):
        # img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
        img = np.round((img.permute(0, 2, 3, 1).cpu().numpy()) * 255)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img


if __name__ == '__main__':
    set_hparams()

    # pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    # cls_name = hparams["trainer_cls"].split(".")[-1]
    # trainer = getattr(importlib.import_module(pkg), cls_name)()

    # global opt
    opt['batch_size'] = 16
    opt['num_workers'] = 8
    opt['num_gpu'] = torch.cuda.device_count()
    opt['dist'] = True


    # swin transformer 的设置
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


    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    trainer = task.fusiondiff.FusionDiffTrainer()
    if not hparams['infer']:
        trainer.train()
    else:
        trainer.test()
