"""
Separation-based baseline: JPEG compression + an IDEAL channel code (one that achieves Shannon capacity).

With k complex channel uses at signal-to-noise ratio SNR, an ideal code delivers at most
    C = k * log2(1 + SNR)  bits                      (AWGN)
    C = k * log2(1 + |h|^2 SNR) bits                  (slow Rayleigh fading, h fixed for the image)
error-free. The transmitter picks the highest JPEG quality whose bitstream fits in C. If even the lowest
quality does not fit, the image is lost and the receiver outputs a flat image at the dataset mean colour.

This baseline is deliberately GENEROUS to separation: real channel codes need a margin below capacity,
and the JPEG header (~600 bytes, larger than many 32x32 payloads) is assumed known at the receiver; only
the entropy-coded payload is counted. Stronger codecs (JPEG2000, BPG) would do better than JPEG.
"""
import io

import numpy as np
from PIL import Image

QUALITIES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95]


def jpeg_payload_bits(buf):
    """Bits after the start-of-scan header and before the end-of-image marker."""
    i = buf.find(b'\xff\xda')
    header_len = int.from_bytes(buf[i + 2:i + 4], 'big')
    return 8 * (len(buf) - (i + 2 + header_len) - 2)


def jpeg_table(images_u8):
    """For every image and quality: payload bits and PSNR of the decoded image.
    images_u8: (N, H, W, 3) uint8 numpy. Returns bits (N, Q) and psnr (N, Q)."""
    n = len(images_u8)
    bits = np.zeros((n, len(QUALITIES)))
    psnrs = np.zeros((n, len(QUALITIES)))
    for i, img in enumerate(images_u8):
        x = img.astype(np.float64) / 255
        for j, q in enumerate(QUALITIES):
            out = io.BytesIO()
            Image.fromarray(img).save(out, format='JPEG', quality=q)
            buf = out.getvalue()
            bits[i, j] = jpeg_payload_bits(buf)
            dec = np.asarray(Image.open(io.BytesIO(buf)).convert('RGB')).astype(np.float64) / 255
            psnrs[i, j] = 10 * np.log10(1 / max(np.mean((x - dec) ** 2), 1e-10))
    return bits, psnrs


def fail_psnr(images_u8, mean_color):
    """Per-image PSNR of a flat mean-colour image (what the receiver shows when the image is lost).
    images_u8: (N, H, W, 3) array or a list of (H, W, 3) arrays of different shapes (Kodak)."""
    return np.array([10 * np.log10(1 / np.mean((im.astype(np.float64) / 255 - mean_color.reshape(1, 1, 3)) ** 2))
                     for im in images_u8])


def separation_psnr(bits, psnrs, fail, k, snr_db, channel='awgn', rng=None):
    """Average PSNR of JPEG + ideal code at one SNR. bits/psnrs from jpeg_table, fail = per-image PSNR
    when the image is lost, k = complex channel uses per image."""
    snr = 10 ** (snr_db / 10)
    n = bits.shape[0]
    if channel == 'awgn':
        capacity = np.full(n, k * np.log2(1 + snr))
    else:
        rng = rng or np.random.default_rng(0)
        h2 = rng.exponential(1.0, size=n)                 # |h|^2 for h ~ CN(0, 1)
        capacity = k * np.log2(1 + h2 * snr)
    fits = bits <= capacity[:, None]                       # (N, Q)
    # quality grows with q, so the best fitting choice is the highest fitting q
    best = np.where(fits.any(1), np.where(fits, psnrs, -np.inf).max(1), fail)
    return float(best.mean())


def cliff_psnr(bits, psnrs, fail, k, design_snr_db, snr_db):
    """The same separation scheme, but designed ONCE for design_snr_db (a fixed JPEG quality and code rate),
    then used at snr_db on an AWGN channel. Above the design SNR nothing improves (the bitstream is fixed);
    below it the code cannot deliver the bits and the image is lost: the "cliff effect"."""
    cap_design = k * np.log2(1 + 10 ** (design_snr_db / 10))
    cap = k * np.log2(1 + 10 ** (snr_db / 10))
    fits = bits <= cap_design
    j = np.where(fits, psnrs, -np.inf).argmax(1)           # quality chosen at the design point
    rows = np.arange(len(bits))
    ok = fits.any(1) & (bits[rows, j] <= cap)
    return float(np.where(ok, psnrs[rows, j], fail).mean())
