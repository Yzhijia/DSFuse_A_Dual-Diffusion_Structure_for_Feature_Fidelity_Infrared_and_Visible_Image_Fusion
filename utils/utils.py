import subprocess
import torch.distributed as dist
import glob
import os
import re
import lpips
import numpy as np
import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn.parallel import DataParallel, DistributedDataParallel
# from .matlab_resize import imresize
# device_ids = [0]
# torch.distributed.init_process_group(backend="nccl")
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
from .hparams import hparams
import torch.nn.functional as F

def random_atheta_generate(batch_size, shape_size, alpha_affine):
    # 对于仿射变换，我们只需要知道变换前的三个点与其对应的变换后的点，就可以通过cv2.getAffineTransform求得变换矩阵.
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    # 初始化 random函数
    random_state = np.random.RandomState(None)

    # pts1 是变换前的三个点，pts2 是变换后的三个点
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    # 进行放射变换, M的类型3*2
    # 将M增加一个维度的batch_size
    # M_s = cv2.getAffineTransform(pts1, pts2)

    a = torch.Tensor([[1., 1., 1./800.], [1., 1., 1./600.]])
    M = torch.tensor(cv2.getAffineTransform(pts1, pts2)).mul(a).unsqueeze(0).expand(batch_size,2,3)
    # print(cv2.getAffineTransform(pts1, pts2))



    # return M
    return move_to_cuda(M)

def random_affine_tensors(img, batch_size):
    theta = random_atheta_generate(batch_size=batch_size, shape_size=(800, 600), alpha_affine=20).view(-1, 2, 3)
    # print(theta)

    # theta = torch.Tensor([[0.707, 0.707, 0], [-0.707, 0.707, 0]]).unsqueeze(dim=0)

    # theta = torch.Tensor([[0.707, 0.707, 0], [-0.707, 0.707, 0]]).unsqueeze(dim=0)
    theta = move_to_cuda(theta)
    grid = F.affine_grid(theta, size=img.shape)
    # x = F.grid_sample(img, grid.to(torch.float32), mode='bilinear',padding_mode='zeros')
    x = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros')
    # print(grid.shape)
    return x


def random_atheta_generate_with_inverse(batch_size, shape_size, alpha_affine):
    # 对于仿射变换，我们只需要知道变换前的三个点与其对应的变换后的点，就可以通过cv2.getAffineTransform求得变换矩阵.
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    # 初始化 random函数
    random_state = np.random.RandomState(None)

    # pts1 是变换前的三个点，pts2 是变换后的三个点
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    # 进行放射变换, M的类型3*2
    # 将M增加一个维度的batch_size
    # M_s = cv2.getAffineTransform(pts1, pts2)

    a = torch.Tensor([[1., 1., 1./800.], [1., 1., 1./600.]])

    M = cv2.getAffineTransform(pts1, pts2)
    ex_row = np.asarray([[0., 0., 1.]], dtype=np.float32)
    M_ex = np.append(M, ex_row, axis=0)
    M_inv = np.linalg.inv(M_ex)[:-1,:]


    M = torch.tensor(M).mul(a).unsqueeze(0).expand(batch_size,2,3)
    M_inv = torch.tensor(M_inv).mul(a).unsqueeze(0).expand(batch_size,2,3)
    # print(cv2.getAffineTransform(pts1, pts2))


    # return (M, M_inv)

    # return M
    return (move_to_cuda(M),move_to_cuda(M_inv))

def random_affine_tensors_with_inverse(img, batch_size):
    (theta, theta_inv) = random_atheta_generate_with_inverse(batch_size=batch_size, shape_size=(800, 600), alpha_affine=40)
    theta = theta.view(-1, 2, 3)
    theta_inv = theta_inv.view(-1, 2, 3)
    # print(theta)

    # theta = torch.Tensor([[0.707, 0.707, 0], [-0.707, 0.707, 0]]).unsqueeze(dim=0)

    # theta = torch.Tensor([[0.707, 0.707, 0], [-0.707, 0.707, 0]]).unsqueeze(dim=0)
    theta = move_to_cuda(theta)
    grid = F.affine_grid(theta, size=img.shape)
    # x = F.grid_sample(img, grid.to(torch.float32), mode='bilinear',padding_mode='zeros')
    # x = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros')

    theta_inv = move_to_cuda(theta_inv)
    grid_inv = F.affine_grid(theta_inv, size=img.shape)

    return (grid, grid_inv)










def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


def tensors_to_np(tensors):
    if isinstance(tensors, dict):
        new_np = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np[k] = v
    elif isinstance(tensors, list):
        new_np = []
        for v in tensors:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np.append(v)
    elif isinstance(tensors, torch.Tensor):
        v = tensors
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if type(v) is dict:
            v = tensors_to_np(v)
        new_np = v
    else:
        raise Exception(f'tensors_to_np does not support type {type(tensors)}.')
    return new_np


def move_to_cpu(tensors):
    ret = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        if type(v) is dict:
            v = move_to_cpu(v)
        ret[k] = v
    return ret


def move_to_cuda(batch, gpu_id=0):
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, 'cuda', None)):
        return batch.cuda(gpu_id, non_blocking=True)
    elif callable(getattr(batch, 'to', None)):
        return batch.to(torch.device('cuda', gpu_id), non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id)
        return batch
    return batch


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))

def load_checkpoint_total(model, optimizer, work_dir):
    checkpoint, _ = get_last_checkpoint(work_dir)
    device_id = torch.device("cuda")
    if checkpoint is not None:

        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model.to(device_ids)
        # model.cuda(device=device_ids[0])
        model.to(device_id)
        model.load_state_dict(checkpoint['state_dict']['model'])
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)

        # model.cuda()
        # print(checkpoint['optimizer_states'])
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])

        # optimizer2.load_state_dict(checkpoint['optimizer_states'][1])
        # optimizer3.load_state_dict(checkpoint['optimizer_states'][2])
        training_step = checkpoint['global_step']
        del checkpoint
        torch.cuda.empty_cache()
    else:
        training_step = 0
        checkpoint_path = os.path.join(work_dir, "initial_weights.pt")

        torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device_id)
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)

        # model.cuda()
    return training_step
def load_checkpoint_total_s(model_ir, model_vi, model_fusion, optimizer_ir, optimizer_vi, optimizer_fusion, work_dir):
    print(work_dir)
    checkpoint, _ = get_last_checkpoint(work_dir)
    device_id = torch.device("cuda")
    if checkpoint is not None:

        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model.to(device_ids)
        # model.cuda(device=device_ids[0])
        model_ir.to(device_id)
        model_vi.to(device_id)
        model_fusion.to(device_id)
        model_ir.load_state_dict(checkpoint['state_dict']['model_ir'])
        model_vi.load_state_dict(checkpoint['state_dict']['model_vi'])
        model_fusion.load_state_dict(checkpoint['state_dict']['model_fusion'])
        model_ir = DistributedDataParallel(model_ir, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        model_vi = DistributedDataParallel(model_vi, device_ids=[torch.cuda.current_device()],
                                           find_unused_parameters=True)
        model_fusion = DistributedDataParallel(model_fusion, device_ids=[torch.cuda.current_device()],
                                           find_unused_parameters=True)
        # model.cuda()
        optimizer_ir.load_state_dict(checkpoint['optimizer_states'][0])
        optimizer_vi.load_state_dict(checkpoint['optimizer_states'][1])
        optimizer_fusion.load_state_dict(checkpoint['optimizer_states'][2])
        training_step = checkpoint['global_step']
        del checkpoint
        torch.cuda.empty_cache()
    else:
        training_step = 0
        # checkpoint_path = os.path.join(work_dir, "initial_weights.ckpt")
        save_checkpoint_total_s(model_ir,model_vi,model_fusion,optimizer_ir,optimizer_vi,optimizer_fusion,work_dir,training_step,hparams['num_ckpt_keep'])
        load_checkpoint_total_s(model_ir, model_vi, model_fusion, optimizer_ir, optimizer_vi, optimizer_fusion,
                                work_dir)
        # torch.save(model.state_dict(), checkpoint_path)
        # dist.barrier()
        # model.load_state_dict(torch.load(checkpoint_path))
        # model.to(device_id)
        # model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
        #                                 find_unused_parameters=True)

        # model.cuda()
    return training_step



def save_checkpoint_total(model, optimizer, work_dir, global_step, num_ckpt_keep):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    ckpt_path = f'{work_dir}/model_ckpt_steps_{global_step}.ckpt'
    print(f'Step@{global_step}: saving model to {ckpt_path}')
    checkpoint = {'global_step': global_step}
    optimizer_states = []
    optimizer_states.append(optimizer.state_dict())
    # optimizer_states.append(optimizer2.state_dict())
    # optimizer_states.append(optimizer3.state_dict())
    checkpoint['optimizer_states'] = optimizer_states
    checkpoint['state_dict'] = {'model': model.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
    for old_ckpt in get_all_ckpts(work_dir)[num_ckpt_keep:]:
        remove_file(old_ckpt)
        print(f'Delete ckpt: {os.path.basename(old_ckpt)}')

def load_checkpoint(model, optimizer1,optimizer2,optimizer3, work_dir):
    checkpoint, _ = get_last_checkpoint(work_dir)
    device_id = torch.device("cuda")
    if checkpoint is not None:

        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model.to(device_ids)
        # model.cuda(device=device_ids[0])
        model.to(device_id)
        model.load_state_dict(checkpoint['state_dict']['model'])
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)

        # model.cuda()
        optimizer1.load_state_dict(checkpoint['optimizer_states'][0])
        optimizer2.load_state_dict(checkpoint['optimizer_states'][1])
        optimizer3.load_state_dict(checkpoint['optimizer_states'][2])
        training_step = checkpoint['global_step']
        del checkpoint
        torch.cuda.empty_cache()
    else:
        training_step = 0
        checkpoint_path = os.path.join(work_dir, "initial_weights.pt")

        torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device_id)
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)

        # model.cuda()
    return training_step


def save_checkpoint(model, optimizer1,optimizer2,optimizer3, work_dir, global_step, num_ckpt_keep):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    ckpt_path = f'{work_dir}/model_ckpt_steps_{global_step}.ckpt'
    print(f'Step@{global_step}: saving model to {ckpt_path}')
    checkpoint = {'global_step': global_step}
    optimizer_states = []
    optimizer_states.append(optimizer1.state_dict())
    optimizer_states.append(optimizer2.state_dict())
    optimizer_states.append(optimizer3.state_dict())
    checkpoint['optimizer_states'] = optimizer_states
    checkpoint['state_dict'] = {'model': model.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
    for old_ckpt in get_all_ckpts(work_dir)[num_ckpt_keep:]:
        remove_file(old_ckpt)
        print(f'Delete ckpt: {os.path.basename(old_ckpt)}')

def save_checkpoint_total_s(model_ir,model_vi, model_fusion, optimizer_ir,optimizer_vi,optimizer_f, work_dir, global_step, num_ckpt_keep):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    ckpt_path = f'{work_dir}/model_ckpt_steps_{global_step}.ckpt'
    print(f'Step@{global_step}: saving model to {ckpt_path}')
    checkpoint = {'global_step': global_step}
    optimizer_states = []
    optimizer_states.append(optimizer_ir.state_dict())
    optimizer_states.append(optimizer_vi.state_dict())
    optimizer_states.append(optimizer_f.state_dict())
    checkpoint['optimizer_states'] = optimizer_states
    checkpoint['state_dict'] = {'model_ir': model_ir.state_dict(),'model_vi':model_vi.state_dict(),'model_fusion':model_fusion.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
    for old_ckpt in get_all_ckpts(work_dir)[num_ckpt_keep:]:
        remove_file(old_ckpt)
        print(f'Delete ckpt: {os.path.basename(old_ckpt)}')


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


def plot_img(img):
    img = img.data.cpu().numpy()
    return np.clip(img, 0, 1)


def load_ckpt(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location='cpu')
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir)
    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        if len([k for k in state_dict.keys() if '.' in k]) > 0:
            state_dict = {k[len(model_name) + 1:]: v for k, v in state_dict.items()
                          if k.startswith(f'{model_name}.')}
        else:
            state_dict = state_dict[model_name]
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        print(f"| load '{model_name}' from '{ckpt_path}'.")
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)


class Measure:
    def __init__(self, net='alex'):
        self.model = lpips.LPIPS(net=net)

    def measure(self, imgA, imgB, img_lr):
        """

        Args:
            imgA: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            imgB: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            img_lr: [C, H, W] uint8  or torch.FloatTensor [-1,1]
            sr_scale:

        Returns: dict of metrics

        """



        if isinstance(imgA, torch.Tensor):
            imgA = np.round((imgA.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
            imgB = np.round((imgB.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
            img_lr = np.round((img_lr.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
        imgA = imgA.transpose(1, 2, 0)
        # imgA_lr = imresize(imgA, 1 / sr_scale)
        imgB = imgB.transpose(1, 2, 0)
        img_lr = img_lr.transpose(1, 2, 0)
        psnr = self.psnr(imgA, imgB)
        # print(imgA.shape)
        # print(imgB.shape)
        ssim = self.ssim(imgA, imgB)
        lpips = self.lpips(imgA, imgB)
        lr_psnr = self.psnr(imgA, img_lr)
        res = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'lr_psnr': lr_psnr}
        return {k: float(v) for k, v in res.items()}

    def lpips(self, imgA, imgB, model=None):
        device = next(self.model.parameters()).device
        tA = t(imgA).to(device)
        tB = t(imgB).to(device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        if imgA.shape[2]==1:
            imgA=np.squeeze(imgA)
        if imgB.shape[2]==1:
            imgB=np.squeeze(imgB)
        score, diff = ssim(imgA, imgB, full=True, multichannel=True, data_range=255)
        return score

    def psnr(self, imgA, imgB):
        return psnr(imgA, imgB, data_range=255)


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1



if __name__ == '__main__':
#     测试图像的随机旋转
    imghdr = cv2.imread(r"H:\code\diffusion\fusion_diffusion\1.jpg",0)
    img = torch.Tensor(cv2.imread(r"H:\code\diffusion\fusion_diffusion\1.jpg",0)).unsqueeze(0).unsqueeze(0)
    # img = img.permute(0, 3, 1, 2)
    imga = random_affine_tensors(img, 1)
    output = imga[0].numpy().transpose(1, 2, 0).squeeze()

    cv2.imwrite(r"H:\code\diffusion\fusion_diffusion\1a.jpg",output)
    # print(img)
    a = []
