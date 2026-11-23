"""
Run every experiment in the study, then evaluate and plot. Models that already exist are skipped, so this can
be stopped and restarted. Total about 1.5-2 hours on an RTX 2050.

  python run_all.py --data_root data
"""
import argparse
import os
import subprocess
import sys

EXPERIMENTS = [
    # RQ1 + RQ2: reproduce the paper (AWGN, fixed training SNR, two bandwidth ratios)
    *[['--c', str(c), '--channel', 'awgn', '--snr', str(s)] for c in (8, 16) for s in (1, 10, 19)],
    # RQ3: slow Rayleigh fading
    *[['--c', '16', '--channel', 'rayleigh', '--snr', str(s)] for s in (1, 10, 19)],
    # RQ4: one model for all SNRs: random training SNR, without and with the SNR as an input
    ['--c', '16', '--channel', 'awgn', '--snr_range', '0', '20'],
    ['--c', '16', '--channel', 'awgn', '--snr_range', '0', '20', '--snr_aware'],
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data')
    p.add_argument('--epochs', type=int, default=120)
    a = p.parse_args()
    from train import main as train_main, run_name
    for exp in EXPERIMENTS:
        args = exp + ['--data_root', a.data_root, '--epochs', str(a.epochs)]
        ns = argparse.Namespace(c=int(exp[1]), channel=exp[3],
                                snr=float(exp[5]) if '--snr' in exp else 10.0,
                                snr_range=[float(exp[5]), float(exp[6])] if '--snr_range' in exp else None,
                                snr_aware='--snr_aware' in exp)
        if os.path.exists(os.path.join('checkpoints', run_name(ns) + '.pt')):
            print(f"skip {run_name(ns)} (already trained)")
            continue
        train_main(args)
    subprocess.run([sys.executable, 'evaluate.py', '--data_root', a.data_root, '--kodak'], check=True)
    subprocess.run([sys.executable, 'tools/plot.py'], check=True)
    subprocess.run([sys.executable, 'tools/show_examples.py', '--data_root', a.data_root], check=True)


if __name__ == '__main__':
    main()
