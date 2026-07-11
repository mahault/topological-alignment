"""
Run all experiments and generate figures for the paper.

Usage:
    python scripts/run_all.py [--exp1] [--exp2] [--exp3] [--figures] [--all]
    python scripts/run_all.py --real --data-dir data/
"""

import sys
import argparse
from pathlib import Path

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))


def run_exp1(data_dir=None):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Synthetic Belief Network")
    print("=" * 60)
    from exp1_synthetic import run_experiment
    results, traj, groups = run_experiment(seed=42, n_steps=10000)

    if data_dir:
        crypto_dir = str(Path(data_dir) / 'crypto')
        if Path(crypto_dir).exists() and list(Path(crypto_dir).glob('*.csv')):
            print("\n--- Crypto Real Data ---")
            from exp1_synthetic import run_experiment_crypto
            crypto_results, assets, regimes = run_experiment_crypto(crypto_dir, seed=42)

    return results, traj, groups


def run_exp2(data_dir=None):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: EEG/Hyperscanning Dyad")
    print("=" * 60)
    from exp2_eeg import run_experiment_synthetic
    results, curv, coupling = run_experiment_synthetic(seed=42)

    if data_dir:
        from data_loaders import (load_collaboration_eeg,
                                  load_parent_child_fnirs,
                                  load_social_touch_fnirs)
        from exp2_eeg import run_experiment_real

        real_datasets = []
        eeg_dir = str(Path(data_dir) / 'eeg_collab')
        fnirs_parent_dir = str(Path(data_dir) / 'fnirs_parent')
        fnirs_touch_dir = str(Path(data_dir) / 'fnirs_touch')

        for dyad_id in range(1, 20):
            d = load_collaboration_eeg(eeg_dir, dyad_id)
            if d is not None:
                real_datasets.append(d)

        for dyad_id in range(1, 65):
            d = load_parent_child_fnirs(fnirs_parent_dir, dyad_id)
            if d is not None:
                real_datasets.append(d)

        for dyad_id in range(1, 50):
            d = load_social_touch_fnirs(fnirs_touch_dir, dyad_id)
            if d is not None:
                real_datasets.append(d)

        if real_datasets:
            print(f"\n--- Real Hyperscanning ({len(real_datasets)} dyads) ---")
            real_results, _ = run_experiment_real(real_datasets)

    return results, curv, coupling


def run_exp3(use_real=False, data_dir=None):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Social Media Belief Basins")
    print("=" * 60)
    from exp3_social_media import run_experiment
    if use_real and data_dir:
        results, communities = run_experiment(
            use_synthetic=False, data_dir=data_dir, seed=42)
    else:
        results, communities = run_experiment(use_synthetic=True, seed=42)
    return results, communities


def run_exp4():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Fisher Geodesics + Gromov-Wasserstein Alignment")
    print("=" * 60)
    from exp4_geodesic_alignment import run_experiment
    results = run_experiment(seed=42)
    return results


def run_figures(use_real=False, data_dir=None):
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    from plotting import generate_all_figures
    generate_all_figures(use_real=use_real, data_dir=data_dir)


def main():
    parser = argparse.ArgumentParser(description='Run belief geodesics experiments')
    parser.add_argument('--exp1', action='store_true', help='Run Experiment 1')
    parser.add_argument('--exp2', action='store_true', help='Run Experiment 2')
    parser.add_argument('--exp3', action='store_true', help='Run Experiment 3')
    parser.add_argument('--exp4', action='store_true',
                        help='Run Experiment 4 (geodesics + GW alignment)')
    parser.add_argument('--figures', action='store_true', help='Generate figures')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--real', action='store_true',
                        help='Use real data alongside synthetic')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Base directory for real data files')
    args = parser.parse_args()

    if args.all or not any([args.exp1, args.exp2, args.exp3, args.exp4,
                            args.figures]):
        args.exp1 = args.exp2 = args.exp3 = args.exp4 = args.figures = True

    data_dir = args.data_dir if args.real else None

    if args.exp1:
        run_exp1(data_dir=data_dir)
    if args.exp2:
        run_exp2(data_dir=data_dir)
    if args.exp3:
        run_exp3(use_real=args.real, data_dir=data_dir)
    if args.exp4:
        run_exp4()
    if args.figures:
        run_figures(use_real=args.real, data_dir=data_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
