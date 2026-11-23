"""Figures from results/eval.json -> results/figures/*.png, and a markdown summary results/summary.md."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('results/eval.json'))
SNRS = R['test_snrs']
RES = R['results']
OUT = 'results/figures'
os.makedirs(OUT, exist_ok=True)


def curve(ax, d, key, **kw):
    if key in d:
        ax.plot(SNRS, [d[key][str(s)] if str(s) in d[key] else d[key][s] for s in SNRS], **kw)


def fig_fixed(dataset, channel, fname, title):
    d = RES[dataset][channel]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, c in zip(axes, (8, 16)):
        for s, col in zip((1, 10, 19), ('tab:blue', 'tab:orange', 'tab:green')):
            curve(ax, d, f'{channel}_c{c}_snr{s}', color=col, marker='o', ms=3, label=f'DeepJSCC, trained at {s} dB')
        curve(ax, d, f'jpeg_c{c}', color='k', ls='--', label='JPEG + ideal code, re-designed for every SNR')
        curve(ax, d, f'jpeg_c{c}_design10', color='gray', ls=':', lw=2, label='JPEG + ideal code, designed for 10 dB')
        ax.set_title(f'{title}, k/n = 1/{96 // c}')
        ax.set_xlabel('test SNR (dB)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('PSNR (dB)')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def fig_adaptive():
    d = RES['cifar10']['awgn']
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for s, col in zip((1, 10, 19), ('tab:blue', 'tab:orange', 'tab:green')):
        curve(ax, d, f'awgn_c16_snr{s}', color=col, lw=1, alpha=0.6, label=f'fixed, trained at {s} dB')
    curve(ax, d, 'awgn_c16_snr0-20', color='tab:gray', marker='s', ms=3, label='one model, random SNR (no SNR input)')
    curve(ax, d, 'awgn_c16_snr0-20_aware', color='tab:red', marker='o', ms=3, label='one model, SNR-aware')
    ax.axvspan(0, 20, color='k', alpha=0.04)
    ax.set_xlabel('test SNR (dB)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('One model for all SNRs? (AWGN, k/n = 1/6)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'snr_aware.png'), dpi=150)
    plt.close(fig)


def fig_kodak_whole():
    """Same models on Kodak: sent as 32x32 blocks (solid) vs the whole image in one pass (dashed)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for s, col in zip((1, 10, 19), ('tab:blue', 'tab:orange', 'tab:green')):
        curve(ax, RES['kodak']['awgn'], f'awgn_c16_snr{s}', color=col, marker='o', ms=3, label=f'trained at {s} dB, 32x32 blocks')
        curve(ax, RES['kodak_whole']['awgn'], f'awgn_c16_snr{s}', color=col, ls='--', label=f'trained at {s} dB, whole image')
    ax.set_xlabel('test SNR (dB)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Kodak, k/n = 1/6: train/test resolution mismatch')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'kodak_whole_vs_blocks.png'), dpi=150)
    plt.close(fig)


def table(dataset, channel, keys, snrs=(0, 4, 10, 16, 20)):
    d = RES[dataset][channel]
    lines = ["| model | " + " | ".join(f"{s} dB" for s in snrs) + " |", "|---" * (len(snrs) + 1) + "|"]
    for k in keys:
        if k in d:
            lines.append(f"| {k} | " + " | ".join(f"{d[k][str(s)]:.2f}" for s in snrs) + " |")
    return "\n".join(lines)


if __name__ == '__main__':
    fig_fixed('cifar10', 'awgn', 'awgn_cifar10.png', 'AWGN, CIFAR-10')
    fig_fixed('cifar10', 'rayleigh', 'rayleigh_cifar10.png', 'Rayleigh fading, CIFAR-10')
    if 'kodak' in RES:
        fig_fixed('kodak', 'awgn', 'awgn_kodak.png', 'AWGN, Kodak in 32x32 blocks')
    if 'kodak_whole' in RES:
        fig_kodak_whole()
    fig_adaptive()
    keys_awgn = [f'awgn_c{c}_snr{s}' for c in (8, 16) for s in (1, 10, 19)] + ['jpeg_c8', 'jpeg_c16', 'jpeg_c8_design10', 'jpeg_c16_design10',
                 'awgn_c16_snr0-20', 'awgn_c16_snr0-20_aware']
    keys_ray = [f'rayleigh_c16_snr{s}' for s in (1, 10, 19)] + ['jpeg_c16']
    parts = ["## CIFAR-10, AWGN (PSNR in dB at each test SNR)\n", table('cifar10', 'awgn', keys_awgn),
             "\n\n## CIFAR-10, slow Rayleigh fading\n", table('cifar10', 'rayleigh', keys_ray)]
    kodak_keys = keys_awgn
    if 'kodak' in RES:
        parts += ["\n\n## Kodak 768x512, AWGN, image sent as 32x32 blocks (models trained on 32x32 CIFAR-10)\n",
                  table('kodak', 'awgn', kodak_keys)]
    if 'kodak_whole' in RES:
        parts += ["\n\n## Kodak, AWGN, whole image in one pass (train/test resolution mismatch, see README)\n",
                  table('kodak_whole', 'awgn', [k for k in kodak_keys if not k.startswith('jpeg')])]
    open('results/summary.md', 'w').write("".join(parts) + "\n")
    print("".join(parts))
    print(f"figures in {OUT}")
