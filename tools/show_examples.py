"""Visual comparison: the same CIFAR-10 test images sent by DeepJSCC and by JPEG + ideal code at several SNRs.
Writes results/figures/examples.png."""
import argparse
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jscc.baselines import QUALITIES, jpeg_payload_bits
from jscc.channel import awgn
from jscc.data import cifar10, to_float
from jscc.model import DeepJSCC


def jpeg_at(img, k, snr_db, mean_color):
    """Best JPEG that fits the ideal-code capacity, or a flat mean-colour image if none fits."""
    cap = k * np.log2(1 + 10 ** (snr_db / 10))
    best = None
    for q in QUALITIES:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='JPEG', quality=q)
        if jpeg_payload_bits(buf.getvalue()) <= cap:
            best = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
    if best is None:
        best = np.broadcast_to((mean_color * 255).astype(np.uint8), img.shape)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data')
    p.add_argument('--ckpt', default='checkpoints/awgn_c16_snr10.pt')
    p.add_argument('--indices', type=int, nargs='+', default=[3, 10, 21, 42])
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = DeepJSCC(ck['args']['c'], ck['args']['snr_aware']).to(dev).eval()
    model.load_state_dict(ck['state_dict'])
    test = cifar10(a.data_root, False, dev)
    mean_color = test.float().mean(dim=(0, 2, 3)).cpu().numpy() / 255
    k = 3 * 32 * 32 * ck['args']['c'] / 96
    snrs = [0, 5, 10, 20]
    torch.manual_seed(0)
    fig, axes = plt.subplots(len(a.indices), 1 + 2 * len(snrs), figsize=(1.3 * (1 + 2 * len(snrs)), 1.45 * len(a.indices)))
    for r, idx in enumerate(a.indices):
        img = test[idx].permute(1, 2, 0).cpu().numpy()
        axes[r, 0].imshow(img)
        for j, s in enumerate(snrs):
            with torch.no_grad():
                x = to_float(test[idx:idx + 1])
                rec = model(x, awgn, torch.tensor([float(s)], device=dev))[0].permute(1, 2, 0).cpu().numpy()
            axes[r, 1 + 2 * j].imshow(rec.clip(0, 1))
            axes[r, 2 + 2 * j].imshow(jpeg_at(img, k, s, mean_color))
            if r == 0:
                axes[r, 1 + 2 * j].set_title(f'DeepJSCC\n{s} dB', fontsize=7)
                axes[r, 2 + 2 * j].set_title(f'JPEG+code\n{s} dB', fontsize=7)
        if r == 0:
            axes[r, 0].set_title('original', fontsize=7)
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle(f"{ck['name']} (k/n = 1/{96 // ck['args']['c']})", fontsize=8)
    fig.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    fig.savefig('results/figures/examples.png', dpi=160)
    print('saved results/figures/examples.png')


if __name__ == '__main__':
    main()
