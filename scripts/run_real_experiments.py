"""
Run all experiments with real data and generate figures.

Usage:
    python scripts/run_real_experiments.py [--data-dir DATA_DIR]
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
RESULTS_FILE = Path(__file__).parent.parent / "real_experiment_results.json"


def run_exp1_crypto(data_dir):
    """Run Experiment 1 with crypto price data."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Crypto Price Data (Real)")
    print("=" * 60)

    from exp1_synthetic import run_experiment_crypto

    crypto_dir = str(Path(data_dir) / "crypto")
    results, assets, regime_indices = run_experiment_crypto(crypto_dir, seed=42)

    # Also run the synthetic experiment for comparison
    print("\n--- Running synthetic baseline for comparison ---")
    from exp1_synthetic import run_experiment as run_exp1_synth
    synth_results, _, _ = run_exp1_synth(seed=42)

    return {
        'crypto': results,
        'crypto_n_assets': len(assets),
        'crypto_regimes': {k: len(v) for k, v in regime_indices.items()},
        'synthetic': synth_results,
    }


def run_exp2_eeg(data_dir):
    """Run Experiment 2 with real EEG data."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: EEG Hyperscanning (Real)")
    print("=" * 60)

    from exp2_eeg import run_experiment_real, run_experiment_synthetic
    from data_loaders import load_collaboration_eeg, HyperscanningData

    eeg_dir = str(Path(data_dir) / "eeg_collab")

    # Try to load real data
    datasets = []
    for dyad_id in range(20):
        data = load_collaboration_eeg(eeg_dir, dyad_id)
        if data is not None:
            datasets.append(data)

    if datasets:
        print(f"Loaded {len(datasets)} real dyads")
        real_results, real_curvatures = run_experiment_real(datasets)
    else:
        print("No real EEG data found, using synthetic only")
        real_results = None

    # Also run synthetic for comparison
    print("\n--- Running synthetic baseline ---")
    synth_results, _, _ = run_experiment_synthetic(seed=42)

    return {
        'real': real_results,
        'synthetic': synth_results,
    }


def run_exp3_reddit(data_dir):
    """Run Experiment 3 with Reddit Politosphere data."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Reddit Politosphere (Real)")
    print("=" * 60)

    from exp3_social_media import run_experiment

    # Check if we have Reddit data
    reddit_dir = Path(data_dir) / "reddit"
    has_reddit = reddit_dir.exists() and any(reddit_dir.glob("*.jsonl"))

    if has_reddit:
        subreddits = []
        for f in reddit_dir.glob("*.jsonl"):
            n_lines = sum(1 for _ in open(str(f), 'r', encoding='utf-8', errors='ignore'))
            if n_lines > 100:
                subreddits.append(f.stem)
                print(f"  r/{f.stem}: {n_lines:,} comments")

        if len(subreddits) >= 3:
            print(f"\nRunning with {len(subreddits)} real subreddits...")
            real_results, real_comms = run_experiment(
                use_synthetic=False, data_dir=str(data_dir),
                subreddits=subreddits, seed=42)
        else:
            print("Not enough subreddits with data, using synthetic")
            real_results = None
            real_comms = None
    else:
        print("No Reddit data found, using synthetic only")
        real_results = None
        real_comms = None

    # Run synthetic baseline
    print("\n--- Running synthetic baseline ---")
    synth_results, synth_comms = run_experiment(use_synthetic=True, seed=42)

    return {
        'real': real_results,
        'synthetic': synth_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = args.data_dir
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    failures = []

    # Run all 3 experiments
    try:
        all_results['exp1'] = run_exp1_crypto(data_dir)
    except Exception as e:
        print(f"\nExp 1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp1'] = {'error': str(e)}
        failures.append('exp1')

    try:
        all_results['exp2'] = run_exp2_eeg(data_dir)
    except Exception as e:
        print(f"\nExp 2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp2'] = {'error': str(e)}
        failures.append('exp2')

    try:
        all_results['exp3'] = run_exp3_reddit(data_dir)
    except Exception as e:
        print(f"\nExp 3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results['exp3'] = {'error': str(e)}
        failures.append('exp3')

    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(str(RESULTS_FILE), 'w') as f:
        json.dump(convert(all_results), f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_FILE}")

    if failures:
        print(f"\nINCOMPLETE RUN: failed experiments: {', '.join(failures)}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if 'exp1' in all_results and 'crypto' in all_results['exp1']:
        cr = all_results['exp1']['crypto']
        print(f"\nExp 1 (Crypto):")
        print(f"  Assets: {all_results['exp1'].get('crypto_n_assets', '?')}")
        print(f"  Regimes: {all_results['exp1'].get('crypto_regimes', '?')}")
        if 'h1_within' in cr:
            ratio = cr['h1_between'] / max(cr['h1_within'], 1e-8)
            print(f"  H1: within={cr['h1_within']:.4f}, between={cr['h1_between']:.4f}, ratio={ratio:.2f}")

    if 'exp2' in all_results:
        e2 = all_results['exp2']
        if e2.get('real') and 'mean_r' in e2['real']:
            print(f"\nExp 2 (Real EEG):")
            print(f"  N dyads: {e2['real']['n_dyads']}")
            print(f"  Mean r: {e2['real']['mean_r']:.3f} +/- {e2['real']['std_r']:.3f}")
        print(f"  Synthetic r: {e2.get('synthetic', {}).get('peak_r', '?')}")

    if 'exp3' in all_results:
        e3 = all_results['exp3']
        if e3.get('real'):
            print(f"\nExp 3 (Reddit):")
            rr = e3['real']
            for key in ['rds_same_type', 'rds_diff_type', 'kl_same_type', 'kl_diff_type']:
                if key in rr:
                    print(f"  {key}: {rr[key]:.4f}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
