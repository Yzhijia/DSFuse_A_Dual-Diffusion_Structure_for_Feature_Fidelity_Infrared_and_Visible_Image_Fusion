import torch
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

theta = torch.Tensor([[0.707, 0.707, 0], [-0.707, 0.707, 0]]).unsqueeze(dim=0)
img = cv2.imread(r"H:\code\diffusion\fusion_diffusion\1.jpg", cv2.IMREAD_GRAYSCALE)
plt.subplot(2, 1, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
img = torch.Tensor(img).unsqueeze(0).unsqueeze(0)
grid = F.affine_grid(theta, size=img.shape)
output = F.grid_sample(img, grid)[0].numpy().transpose(1, 2, 0).squeeze()
plt.subplot(2, 1, 2)
plt.imshow(output, cmap='gray')
plt.axis('off')
plt.show()

a = []