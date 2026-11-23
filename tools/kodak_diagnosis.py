"""
Why does a model trained on 32x32 CIFAR-10 fail on whole 768x512 Kodak images, and why does sending the image
as 32x32 blocks fix it? Each row changes ONE thing between the two settings.

  python tools/kodak_diagnosis.py --data_root data                    -> results/kodak_diagnosis.md

Finding (README, "Generalising to Kodak"): the failure is there even with NO channel noise, power allocation
is not the cause, and both the encoder and the decoder have to see 32x32 frames. The encoder's receptive field
(61 px) is larger than a CIFAR image, so in training every latent value was computed from a window that
included the zero-padded border. The learned code depends on that border; latents from the interior of a
large image never occurred in training.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jscc.channel import awgn
from jscc.data import kodak, psnr, to_float
from jscc.model import DeepJSCC, power_normalize


def split(x, t):
    """(1, C, H, W) -> (H/t * W/t, C, t, t) non-overlapping blocks."""
    _, c, h, w = x.shape
    return x.unfold(2, t, t).unfold(3, t, t).permute(0, 2, 3, 1, 4, 5).reshape(-1, c, t, t), (h, w)


def merge(p, hw, t):
    h, w = hw
    return p.view(1, h // t, w // t, p.shape[1], t, t).permute(0, 3, 1, 4, 2, 5).reshape(1, p.shape[1], h, w)


def receptive_field():
    """Encoder: two 5x5 stride-2 convs, then three 5x5 stride-1 convs at stride 4."""
    rf, jump = 1, 1
    for k, s in [(5, 2), (5, 2), (5, 1), (5, 1), (5, 1)]:
        rf += (k - 1) * jump
        jump *= s
    return rf


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data')
    p.add_argument('--ckpt', default='checkpoints/awgn_c16_snr19.pt')
    p.add_argument('--snr', type=float, default=20.0)
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = DeepJSCC(ck['args']['c'], ck['args']['snr_aware']).to(dev).eval()
    m.load_state_dict(ck['state_dict'])
    images = [to_float(im.unsqueeze(0).to(dev)) for im in kodak(a.data_root)]
    snr = lambda n: torch.full((n,), a.snr, device=dev)
    enc, dec = m.encoder, m.decoder

    def whole(x):                         # the whole image in one pass
        return dec(awgn(power_normalize(enc(x)), snr(1)))

    def whole_noiseless(x):
        return dec(power_normalize(enc(x)))

    def block_power(x):                   # whole encoder + decoder, but power normalised per 32x32 region
        z, hw = split(enc(x), 8)
        return dec(merge(awgn(power_normalize(z), snr(len(z))), hw, 8))

    def tiled_encoder(x):                 # encoder sees 32x32 tiles, decoder the stitched latent
        t, hw = split(x, 32)
        z = awgn(power_normalize(enc(t)), snr(len(t)))
        return dec(merge(z, (hw[0] // 4, hw[1] // 4), 8))

    def tiled_decoder(x):                 # encoder sees the whole image, decoder decodes 8x8 latent tiles
        z, hw = split(awgn(power_normalize(enc(x)), snr(1)), 8)
        return merge(dec(z), (hw[0] * 4, hw[1] * 4), 32)

    def tiled(x):                         # both on 32x32 blocks = the setting used for the Kodak results
        t, hw = split(x, 32)
        return merge(m(t, awgn, snr(len(t))), hw, 32)

    rows = [('whole image in one pass', whole),
            ('whole image, no channel noise at all', whole_noiseless),
            ('whole image, power normalised per 32x32 region', block_power),
            ('encoder on 32x32 tiles, decoder on the whole latent', tiled_encoder),
            ('encoder on the whole image, decoder on 8x8 latent tiles', tiled_decoder),
            ('encoder and decoder on 32x32 tiles', tiled)]
    torch.manual_seed(0)
    lines = [f"Model `{ck['name']}`, AWGN at {a.snr:g} dB, mean PSNR over the 24 Kodak images. "
             f"Encoder receptive field: {receptive_field()} px (CIFAR images are 32 px).\n",
             "| setting | PSNR (dB) |", "|---|---|"]
    for name, f in rows:
        v = torch.stack([psnr(x, f(x)) for x in images]).mean().item()
        lines.append(f"| {name} | {v:.2f} |")
        print(lines[-1])
    os.makedirs('results', exist_ok=True)
    open('results/kodak_diagnosis.md', 'w').write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
