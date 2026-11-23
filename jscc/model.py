"""
DeepJSCC (Bourtsoulatze, Kurka & Gunduz, "Deep Joint Source-Channel Coding for Wireless Image Transmission",
IEEE TCCN 2019 / ICASSP 2019): an autoencoder whose bottleneck IS the channel input.

    image x (3 x H x W) --encoder--> z (c x H/4 x W/4 real values = k complex channel symbols)
        --power normalisation--> --noisy channel--> y --decoder--> x_hat

Bandwidth ratio  k / n = (c * H/4 * W/4 / 2) / (3 * H * W) = c / 96.
  c = 8  ->  k/n = 1/12        c = 16 ->  k/n = 1/6       (the two ratios in the paper)

Layers follow the paper: 5x5 convolutions, PReLU, two stride-2 layers in the encoder and two stride-2
transposed convolutions in the decoder, sigmoid output.

Extension (`snr_aware=True`): the channel SNR is appended as an extra constant feature map at the encoder's
last layer and the decoder's first layer, so ONE model can adapt to any SNR (instead of one model per SNR).
This is a simplified version of the attention-based ADJSCC idea (Xu et al., IEEE TCSVT 2021).
"""
import torch
import torch.nn as nn


def conv(cin, cout, stride):
    return nn.Conv2d(cin, cout, 5, stride=stride, padding=2)


def deconv(cin, cout, stride):
    return nn.ConvTranspose2d(cin, cout, 5, stride=stride, padding=2, output_padding=stride - 1)


class Encoder(nn.Module):
    def __init__(self, c, snr_aware=False):
        super().__init__()
        self.snr_aware = snr_aware
        self.body = nn.Sequential(
            conv(3, 16, 2), nn.PReLU(),       # H/2
            conv(16, 32, 2), nn.PReLU(),      # H/4
            conv(32, 32, 1), nn.PReLU(),
            conv(32, 32, 1), nn.PReLU(),
        )
        self.out = conv(32 + int(snr_aware), c, 1)   # no activation: the power normalisation follows

    def forward(self, x, snr_db=None):
        h = self.body(x)
        if self.snr_aware:
            h = torch.cat([h, _snr_map(h, snr_db)], dim=1)
        return self.out(h)


class Decoder(nn.Module):
    def __init__(self, c, snr_aware=False):
        super().__init__()
        self.snr_aware = snr_aware
        self.inp = deconv(c + int(snr_aware), 32, 1)
        self.body = nn.Sequential(
            nn.PReLU(),
            deconv(32, 32, 1), nn.PReLU(),
            deconv(32, 32, 1), nn.PReLU(),
            deconv(32, 16, 2), nn.PReLU(),    # H/2
            deconv(16, 3, 2), nn.Sigmoid(),   # H, pixel values in [0, 1]
        )

    def forward(self, y, snr_db=None):
        if self.snr_aware:
            y = torch.cat([y, _snr_map(y, snr_db)], dim=1)
        return self.body(self.inp(y))


def _snr_map(h, snr_db):
    """Constant feature map holding the SNR (scaled to roughly [0, 1]), one value per image."""
    s = torch.as_tensor(snr_db, dtype=h.dtype, device=h.device).reshape(-1)
    if s.numel() == 1:
        s = s.expand(h.shape[0])
    return (s.view(-1, 1, 1, 1) / 20.0).expand(h.shape[0], 1, h.shape[2], h.shape[3])


def power_normalize(z):
    """Scale each image's symbols so the average power per complex channel symbol is 1:
    with k complex symbols (2k real values), ||z||^2 = k."""
    flat = z.flatten(1)
    k = flat.shape[1] / 2
    flat = flat * (k ** 0.5) / flat.norm(dim=1, keepdim=True)
    return flat.view_as(z)


class DeepJSCC(nn.Module):
    def __init__(self, c=16, snr_aware=False):
        super().__init__()
        self.c, self.snr_aware = c, snr_aware
        self.encoder = Encoder(c, snr_aware)
        self.decoder = Decoder(c, snr_aware)

    def forward(self, x, channel, snr_db):
        z = power_normalize(self.encoder(x, snr_db))
        y = channel(z, snr_db)
        return self.decoder(y, snr_db)

    @staticmethod
    def bandwidth_ratio(c):
        return c / 96
