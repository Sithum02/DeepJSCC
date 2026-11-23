"""
Train one DeepJSCC model on CIFAR-10.

  python train.py --c 16 --channel awgn --snr 10                 # fixed training SNR (as in the paper)
  python train.py --c 16 --channel awgn --snr_range 0 20         # random SNR per image, no SNR input
  python train.py --c 16 --channel awgn --snr_range 0 20 --snr_aware   # random SNR + SNR given to the model

Saves checkpoints/<name>.pt (weights + settings + training log). About 5-10 minutes on an RTX 2050.
"""
import argparse
import json
import math
import os
import time

import torch

from jscc.channel import CHANNELS
from jscc.data import cifar10, psnr, to_float
from jscc.model import DeepJSCC


def run_name(a):
    snr = f"snr{a.snr:g}" if a.snr_range is None else f"snr{a.snr_range[0]:g}-{a.snr_range[1]:g}"
    return f"{a.channel}_c{a.c}_{snr}{'_aware' if a.snr_aware else ''}"


def sample_snr(a, n, device):
    if a.snr_range is None:
        return torch.full((n,), float(a.snr), device=device)
    lo, hi = a.snr_range
    return torch.rand(n, device=device) * (hi - lo) + lo


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--c', type=int, default=16, help='latent channels: 8 -> k/n = 1/12, 16 -> k/n = 1/6')
    p.add_argument('--channel', choices=list(CHANNELS), default='awgn')
    p.add_argument('--snr', type=float, default=10.0, help='training SNR in dB')
    p.add_argument('--snr_range', type=float, nargs=2, default=None, help='train with SNR ~ U(lo, hi) dB')
    p.add_argument('--snr_aware', action='store_true', help='feed the SNR to encoder and decoder')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_root', default='data')
    p.add_argument('--out', default='checkpoints')
    a = p.parse_args(argv)
    assert not (a.snr_aware and a.snr_range is None), "--snr_aware only makes sense with --snr_range"

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train = cifar10(a.data_root, True, dev)
    val = cifar10(a.data_root, False, dev)[:2000]
    model = DeepJSCC(a.c, a.snr_aware).to(dev)
    channel = CHANNELS[a.channel]
    steps_per_epoch = len(train) // a.batch_size
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * steps_per_epoch, eta_min=a.lr / 100)
    name = run_name(a)
    log = []
    print(f"{name}: k/n = 1/{96 // a.c}, {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters, {dev}")

    start = time.time()
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(len(train), device=dev)
        total = 0.0
        for s in range(steps_per_epoch):
            x = to_float(train[perm[s * a.batch_size:(s + 1) * a.batch_size]])
            flip = torch.rand(x.shape[0], device=dev) < 0.5                 # random horizontal flips
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
            snr = sample_snr(a, x.shape[0], dev)
            loss = torch.mean((model(x, channel, snr) - x) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            xv = to_float(val)
            eval_snr = a.snr if a.snr_range is None else sum(a.snr_range) / 2
            val_psnr = psnr(xv, model(xv, channel, torch.full((len(xv),), eval_snr, device=dev))).mean().item()
        train_psnr = 10 * math.log10(1 / (total / steps_per_epoch))
        log.append(dict(epoch=epoch + 1, train_psnr=train_psnr, val_psnr=val_psnr))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}  train {train_psnr:5.2f} dB  val {val_psnr:5.2f} dB "
                  f"(at {eval_snr:g} dB)  {(time.time() - start) / 60:.1f} min")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{name}.pt")
    torch.save(dict(state_dict=model.state_dict(), args=vars(a), log=log, name=name), path)
    print(f"saved {path} ({(time.time() - start) / 60:.1f} min)")
    return path


if __name__ == '__main__':
    main()
