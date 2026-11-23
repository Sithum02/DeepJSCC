"""CIFAR-10 kept entirely on the GPU as uint8 (150 MB), so training is not limited by data loading,
plus the 24-image Kodak set (768x512) for testing on larger images."""
import os
import urllib.request

import numpy as np
import torch
from PIL import Image


def cifar10(root, train, device):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()      # (N, 3, 32, 32) uint8
    return x.to(device)


def to_float(batch_u8):
    return batch_u8.float() / 255.0


def kodak(root):
    """24 Kodak PNGs as a list of (3, H, W) uint8 tensors. All are 768x512 or 512x768."""
    folder = os.path.join(root, 'kodak')
    os.makedirs(folder, exist_ok=True)
    images = []
    for i in range(1, 25):
        path = os.path.join(folder, f'kodim{i:02d}.png')
        if not os.path.exists(path):
            urllib.request.urlretrieve(f'https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png', path)
        images.append(torch.from_numpy(np.asarray(Image.open(path).convert('RGB'))).permute(2, 0, 1).contiguous())
    return images


def psnr(x, x_hat):
    """Per-image PSNR in dB for images in [0, 1]; returns a (B,) tensor."""
    mse = ((x - x_hat) ** 2).flatten(1).mean(dim=1).clamp(min=1e-10)
    return 10 * torch.log10(1.0 / mse)
