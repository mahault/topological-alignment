"""
Precompute SBERT embeddings for Reddit Politosphere subreddits.
===============================================================

Run this script on a machine with a working PyTorch + sentence-transformers
installation. Saves .npy files in data/embeddings/ that the main pipeline
can load without requiring torch at runtime.

Usage:
    python scripts/precompute_embeddings.py --data-dir data/reddit --out-dir data/embeddings
    python scripts/precompute_embeddings.py --data-dir data/reddit --subreddits socialism Conservative
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))


DEFAULT_SUBREDDITS = [
    'socialism', 'LateStageCapitalism', 'ChapoTrapHouse',
    'Conservative', 'The_Donald', 'Republican',
    'PoliticalDiscussion', 'NeutralPolitics', 'moderatepolitics',
    'politics',
]


def precompute(data_dir: str, out_dir: str, subreddits: list[str],
               model_name: str = 'all-mpnet-base-v2',
               max_posts: int = 50000, batch_size: int = 128):
    """Load texts from Reddit data files, embed with SBERT, save as .npy."""
    from data_loaders import load_reddit_politosphere

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load model once
    print(f"Loading SBERT model: {model_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    for sub in subreddits:
        out_file = out_path / f"{sub}_embeddings.npy"
        if out_file.exists():
            existing = np.load(str(out_file))
            print(f"  {sub}: already exists ({existing.shape[0]} embeddings), skipping")
            continue

        print(f"\nProcessing r/{sub}...")
        data = load_reddit_politosphere(data_dir, sub, max_posts=max_posts)

        if not data.texts:
            print(f"  No texts found for r/{sub}, skipping")
            continue

        print(f"  Loaded {len(data.texts)} texts")
        print(f"  Embedding with {model_name}...")

        embeddings = model.encode(
            data.texts, batch_size=batch_size,
            show_progress_bar=True, normalize_embeddings=False,
        )

        np.save(str(out_file), embeddings)
        print(f"  Saved {embeddings.shape} to {out_file}")

        # Also save timestamps if available
        if data.timestamps is not None and len(data.timestamps) > 0:
            ts_file = out_path / f"{sub}_timestamps.npy"
            np.save(str(ts_file), data.timestamps)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description='Precompute SBERT embeddings for Reddit subreddits')
    parser.add_argument('--data-dir', type=str, default='data/reddit',
                        help='Directory with Reddit JSONL/BZ2 files')
    parser.add_argument('--out-dir', type=str, default='data/embeddings',
                        help='Output directory for .npy embeddings')
    parser.add_argument('--subreddits', nargs='+', default=None,
                        help='Subreddits to process (default: all 10)')
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2',
                        help='SBERT model name')
    parser.add_argument('--max-posts', type=int, default=50000,
                        help='Maximum posts per subreddit')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Encoding batch size')
    args = parser.parse_args()

    subs = args.subreddits if args.subreddits else DEFAULT_SUBREDDITS
    precompute(args.data_dir, args.out_dir, subs,
               model_name=args.model, max_posts=args.max_posts,
               batch_size=args.batch_size)


if __name__ == '__main__':
    main()
