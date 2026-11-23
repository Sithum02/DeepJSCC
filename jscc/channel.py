"""
Channel models. The encoder output z (B, c, h, w) is read as k = c*h*w/2 complex symbols: the first half of
the values are the real parts, the second half the imaginary parts. Average symbol power is 1 (see
power_normalize), so SNR = 1 / noise variance.

AWGN:      y = z + n,              n ~ CN(0, sigma^2),  sigma^2 = 10^(-SNR_dB / 10)
Rayleigh:  y = h z + n,            h ~ CN(0, 1), one value per image (slow / block fading)
           The receiver is assumed to know h (perfect channel state information) and equalises: y / h.
           A deep fade (small |h|) therefore amplifies the noise for that whole image.
"""
import torch


def _noise_std(snr_db, like):
    snr_db = torch.as_tensor(snr_db, dtype=like.dtype, device=like.device).reshape(-1)
    if snr_db.numel() == 1:
        snr_db = snr_db.expand(like.shape[0])
    sigma2 = 10 ** (-snr_db / 10)
    return (sigma2 / 2).sqrt().view(-1, *([1] * (like.dim() - 1)))   # per real component


def awgn(z, snr_db):
    return z + torch.randn_like(z) * _noise_std(snr_db, z)


def rayleigh(z, snr_db):
    b = z.shape[0]
    flat = z.flatten(1)
    k = flat.shape[1] // 2
    zc = torch.complex(flat[:, :k], flat[:, k:])
    h = torch.complex(torch.randn(b, 1, device=z.device), torch.randn(b, 1, device=z.device)) / 2 ** 0.5
    std = _noise_std(snr_db, z).view(b, 1)
    n = torch.complex(torch.randn_like(flat[:, :k]), torch.randn_like(flat[:, :k])) * std
    y = (h * zc + n) / h                                   # zero-forcing equalisation with perfect CSI
    return torch.cat([y.real, y.imag], dim=1).view_as(z)


CHANNELS = {'awgn': awgn, 'rayleigh': rayleigh}
