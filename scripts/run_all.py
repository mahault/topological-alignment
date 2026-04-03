"""
Run all experiments and generate figures for the paper.

Usage:
    python scripts/run_all.py [--exp1] [--exp2] [--exp3] [--figures] [--all]
"""

import sys
import argparse
from pathlib import Path

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))


def run_exp1():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Synthetic Belief Network")
    print("=" * 60)
    from exp1_synthetic import run_experiment
    results, traj, groups = run_experiment(seed=42, n_steps=10000)
    return results, traj, groups


def run_exp2():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: EEG/Hyperscanning Dyad")
    print("=" * 60)
    from exp2_eeg import run_experiment_synthetic
    results, curv, coupling = run_experiment_synthetic(seed=42)
    return results, curv, coupling


def run_exp3():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Social Media Belief Basins")
    print("=" * 60)
    from exp3_social_media import run_experiment
    results, communities = run_experiment(use_synthetic=True, seed=42)
    return results, communities


def run_figures():
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    from plotting import generate_all_figures
    generate_all_figures()


def main():
    parser = argparse.ArgumentParser(description='Run belief geodesics experiments')
    parser.add_argument('--exp1', action='store_true', help='Run Experiment 1')
    parser.add_argument('--exp2', action='store_true', help='Run Experiment 2')
    parser.add_argument('--exp3', action='store_true', help='Run Experiment 3')
    parser.add_argument('--figures', action='store_true', help='Generate figures')
    parser.add_argument('--all', action='store_true', help='Run everything')
    args = parser.parse_args()

    if args.all or not any([args.exp1, args.exp2, args.exp3, args.figures]):
        args.exp1 = args.exp2 = args.exp3 = args.figures = True

    if args.exp1:
        run_exp1()
    if args.exp2:
        run_exp2()
    if args.exp3:
        run_exp3()
    if args.figures:
        run_figures()

    print("\nDone.")


if __name__ == '__main__':
    main()
