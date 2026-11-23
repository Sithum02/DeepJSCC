"""
Sweep the test SNR for every trained model and for the JPEG + ideal-code baseline.

  python evaluate.py                       # all checkpoints in checkpoints/, AWGN and Rayleigh, CIFAR-10 test set
  python evaluate.py --kodak               # also test the AWGN models on the 24 Kodak images (768x512)

Writes results/eval.json:  {dataset: {channel: {model_name or 'jpeg_c8'/'jpeg_c16': {snr: PSNR}}}}
Every model is tested on the channel it was trained on; the baseline on both channels.

Kodak is tested two ways:
  'kodak'        the image is sent as 384 independent 32x32 blocks (one packet each, each with its own power
                 constraint), the size the models were trained on. PSNR is over the whole reassembled image.
  'kodak_whole'  the whole 768x512 image in one pass. Models trained only on 32x32 images do badly here, even
                 with no channel noise: the encoder's receptive field (61 px) is larger than a CIFAR image, so
                 during training every latent saw the zero-padded border and the learned code depends on it.
                 Interior positions of a large image are out of distribution. (The paper trained on larger
                 ImageNet crops and did not have this problem.) See README, "Generalising to Kodak".
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from jscc.baselines import cliff_psnr, fail_psnr, jpeg_table, separation_psnr
from jscc.channel import CHANNELS
from jscc.data import cifar10, kodak, psnr, to_float
from jscc.model import DeepJSCC

TEST_SNRS = list(range(-4, 27, 2))


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = DeepJSCC(ck['args']['c'], ck['args']['snr_aware']).to(dev).eval()
    m.load_state_dict(ck['state_dict'])
    return m, ck


def to_tiles(x, t=32):
    """(3, H, W) -> (H/t * W/t, 3, t, t): non-overlapping t x t blocks."""
    c = x.shape[0]
    return x.unfold(1, t, t).unfold(2, t, t).permute(1, 2, 0, 3, 4).reshape(-1, c, t, t)


def batches(images, dev, tiles=False, batch=500):
    """CIFAR: one (N, 3, 32, 32) tensor -> batches. Kodak: list of images of different shapes -> one image per
    batch, either whole (1, 3, H, W) or as its 32x32 blocks (384, 3, 32, 32)."""
    if torch.is_tensor(images):
        for i in range(0, len(images), batch):
            yield to_float(images[i:i + batch].to(dev))
    else:
        for im in images:
            x = to_float(im.to(dev))
            yield to_tiles(x) if tiles else x.unsqueeze(0)


@torch.no_grad()
def model_curve(model, images, channel, dev, tiles=False, seed=0):
    torch.manual_seed(seed)
    out = {}
    for snr in TEST_SNRS:
        vals = []
        for x in batches(images, dev, tiles):
            x_hat = model(x, CHANNELS[channel], torch.full((len(x),), float(snr), device=dev))
            if tiles:   # the blocks of ONE image: PSNR of the reassembled image (= mean MSE over its blocks)
                vals.append(psnr(x.reshape(1, -1), x_hat.reshape(1, -1)).cpu())
            else:
                vals.append(psnr(x, x_hat).cpu())
        out[snr] = torch.cat(vals).mean().item()
    return out


def baseline_curves(images_np, mean_color, n_pixels_values):
    bits, ps = jpeg_table(images_np)
    fail = fail_psnr(images_np, mean_color)
    res = {}
    for channel in ['awgn', 'rayleigh']:
        res[channel] = {}
        for c in (8, 16):
            k = n_pixels_values * c / 96                    # complex channel uses for this bandwidth ratio
            res[channel][f'jpeg_c{c}'] = {snr: separation_psnr(bits, ps, fail, k, snr, channel,
                                                                np.random.default_rng(0)) for snr in TEST_SNRS}
            if channel == 'awgn':   # the same scheme frozen at a 10 dB design point: shows the cliff effect
                res[channel][f'jpeg_c{c}_design10'] = {snr: cliff_psnr(bits, ps, fail, k, 10, snr) for snr in TEST_SNRS}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', default='checkpoints')
    p.add_argument('--data_root', default='data')
    p.add_argument('--kodak', action='store_true')
    p.add_argument('--out', default='results')
    p.add_argument('--baselines_only', action='store_true',
                   help='recompute only the JPEG baselines (CPU) and merge them into an existing eval.json')
    a = p.parse_args()
    dev = 'cpu' if a.baselines_only or not torch.cuda.is_available() else 'cuda'
    paths = [] if a.baselines_only else sorted(glob.glob(os.path.join(a.ckpts, '*.pt')))
    out_path = os.path.join(a.out, 'eval.json')
    results = json.load(open(out_path))['results'] if a.baselines_only else {}

    test = cifar10(a.data_root, False, dev)                # (10000, 3, 32, 32)
    datasets = {'cifar10': (test, False)}
    if a.kodak:
        datasets['kodak'] = (kodak(a.data_root), True)
        datasets['kodak_whole'] = (datasets['kodak'][0], False)
    for name, (images, tiles) in datasets.items():
        results.setdefault(name, {'awgn': {}, 'rayleigh': {}})
        for path in paths:
            model, ck = load(path, dev)
            ch = ck['args']['channel']
            if name != 'cifar10' and ch != 'awgn':
                continue            # Kodak: AWGN models only (blockwise Rayleigh would be a different channel)
            results[name][ch][ck['name']] = model_curve(model, images, ch, dev, tiles)
            print(f"{name:11s} {ck['name']:<28s} " + ' '.join(f"{v:5.1f}" for v in results[name][ch][ck['name']].values()))
        if name == 'kodak_whole':
            continue
        if name == 'cifar10':
            imgs_np, n_values = test.permute(0, 2, 3, 1).cpu().numpy(), 3 * 32 * 32
        else:   # every Kodak image has 768 * 512 pixels (landscape or portrait)
            imgs_np, n_values = [im.permute(1, 2, 0).numpy() for im in images], 3 * 768 * 512
        mean_color = test.float().mean(dim=(0, 2, 3)).cpu().numpy() / 255
        base = baseline_curves(imgs_np, mean_color, n_values)
        for ch in base:
            results[name][ch].update(base[ch])
            for k, v in base[ch].items():
                print(f"{name:11s} {ch:8s} {k:<19s} " + ' '.join(f"{x:5.1f}" for x in v.values()))

    os.makedirs(a.out, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(dict(test_snrs=TEST_SNRS, results=results), f, indent=1)
    print(f"saved {out_path}")


if __name__ == '__main__':
    main()
