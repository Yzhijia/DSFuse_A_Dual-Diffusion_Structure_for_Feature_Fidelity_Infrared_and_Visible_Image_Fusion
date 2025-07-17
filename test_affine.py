import cv2
import numpy as np
import torch
import torch.nn.functional as F


# M = np.asarray([[0.707, -0.707,0],[0.707,0.707,0]],dtype=np.float32)
# ex_row = np.asarray([[0,0,1]],dtype=np.float32)
# M_ex = np.append(M,ex_row, axis=0)
# # y = torch.inverse(M)
# print(np.linalg.inv(M_ex))


def move_to_cuda(batch, gpu_id=1):
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

    a = torch.Tensor([[1., 1., 1./640.], [1., 1., 1./480.]])

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

def random_affine_tensors_with_inverse(shape, batch_size):
    (theta, theta_inv) = random_atheta_generate_with_inverse(batch_size=batch_size, shape_size=(640, 480),
                                                             alpha_affine=40)
    theta = theta.view(-1, 2, 3)
    theta_inv = theta_inv.view(-1, 2, 3)
    # print(theta)

    # theta = torch.Tensor([[0.707, 0.707, 0], [-0.707, 0.707, 0]]).unsqueeze(dim=0)

    # theta = torch.Tensor([[0.707, 0.707, 0], [-0.707, 0.707, 0]]).unsqueeze(dim=0)
    theta = move_to_cuda(theta)
    grid = F.affine_grid(theta, size=shape)
    # x = F.grid_sample(img, grid.to(torch.float32), mode='bilinear',padding_mode='zeros')
    # x = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros')

    theta_inv = move_to_cuda(theta_inv)
    grid_inv = F.affine_grid(theta_inv, size=shape)

    return (grid, grid_inv)





(M, M_inv) = random_affine_tensors_with_inverse(shape=(1,640, 480),batch_size=1)
print(M.shape)
print(M_inv.shape)









