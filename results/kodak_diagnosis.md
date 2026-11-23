Model `awgn_c16_snr19`, AWGN at 20 dB, mean PSNR over the 24 Kodak images. Encoder receptive field: 61 px (CIFAR images are 32 px).

| setting | PSNR (dB) |
|---|---|
| whole image in one pass | 23.28 |
| whole image, no channel noise at all | 23.39 |
| whole image, power normalised per 32x32 region | 24.37 |
| encoder on 32x32 tiles, decoder on the whole latent | 26.29 |
| encoder on the whole image, decoder on 8x8 latent tiles | 22.13 |
| encoder and decoder on 32x32 tiles | 35.01 |
