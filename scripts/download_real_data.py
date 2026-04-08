"""
Download real-world datasets for all 3 experiments.
Run on spectrum (or any Linux machine with internet access).

Usage:
    python scripts/download_real_data.py --all
    python scripts/download_real_data.py --crypto
    python scripts/download_real_data.py --reddit
    python scripts/download_real_data.py --eeg
"""

import os
import sys
import json
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


# ====================================================================
# Experiment 1: Crypto price data via yfinance
# ====================================================================

CRYPTO_ASSETS = {
    # Stablecoins / low-vol (→ "stable" regime)
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    # Large-cap (→ "large_cap" regime)
    'BNB-USD': 'BNB',
    'XRP-USD': 'Ripple',
    'ADA-USD': 'Cardano',
    'SOL-USD': 'Solana',
    'DOT-USD': 'Polkadot',
    'AVAX-USD': 'Avalanche',
    'MATIC-USD': 'Polygon',
    'LINK-USD': 'Chainlink',
    # Mid/small-cap (→ "mid_cap" regime)
    'DOGE-USD': 'Dogecoin',
    'SHIB-USD': 'Shiba Inu',
    'UNI-USD': 'Uniswap',
    'AAVE-USD': 'Aave',
    'FIL-USD': 'Filecoin',
    'SAND-USD': 'Sandbox',
    'MANA-USD': 'Decentraland',
    'AXS-USD': 'Axie Infinity',
    'ALGO-USD': 'Algorand',
    'FTM-USD': 'Fantom',
}


def download_crypto():
    """Download crypto price data using yfinance."""
    import yfinance as yf
    import pandas as pd

    out_dir = DATA_DIR / "crypto"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Downloading crypto price data...")
    print(f"{'='*60}")

    for ticker, name in CRYPTO_ASSETS.items():
        out_file = out_dir / f"{name}.csv"
        if out_file.exists():
            print(f"  {name}: already exists, skipping")
            continue

        print(f"  Downloading {name} ({ticker})...")
        try:
            data = yf.download(ticker, period="max", interval="1d",
                               progress=False, auto_adjust=True)
            if len(data) < 30:
                print(f"    Only {len(data)} rows, skipping")
                continue

            # Save in CryptoDataDownload-compatible format
            df = pd.DataFrame({
                'Date': data.index.strftime('%Y-%m-%d'),
                'Open': data['Open'].values.flatten(),
                'High': data['High'].values.flatten(),
                'Low': data['Low'].values.flatten(),
                'Close': data['Close'].values.flatten(),
                'Volume': data['Volume'].values.flatten(),
            })
            df.to_csv(str(out_file), index=False)
            print(f"    Saved {len(df)} days to {out_file.name}")
        except Exception as e:
            print(f"    Failed: {e}")

    print(f"\nCrypto data in: {out_dir}")
    print(f"Files: {len(list(out_dir.glob('*.csv')))}")


# ====================================================================
# Experiment 3: Reddit Politosphere
# ====================================================================

TARGET_SUBREDDITS = {
    'socialism', 'LateStageCapitalism', 'ChapoTrapHouse',
    'Conservative', 'The_Donald', 'Republican',
    'PoliticalDiscussion', 'NeutralPolitics', 'moderatepolitics',
    'politics',
}

# Download 3 months from different years for temporal coverage
REDDIT_MONTHS = [
    'comments_2016-06.bz2',  # election year, high activity
    'comments_2018-01.bz2',  # post-election, all subs active
    'comments_2019-06.bz2',  # pre-ban era
]

ZENODO_RECORD = "5851729"
ZENODO_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"


def download_reddit():
    """Download Reddit Politosphere data from Zenodo and filter by subreddit."""
    import bz2
    import urllib.request

    reddit_dir = DATA_DIR / "reddit"
    reddit_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Downloading Reddit Politosphere data...")
    print(f"{'='*60}")

    # Check if we already have per-subreddit files
    existing = {f.stem for f in reddit_dir.glob("*.jsonl")}
    if TARGET_SUBREDDITS.issubset(existing):
        print("  All subreddit files already exist, skipping download")
        for sub in sorted(TARGET_SUBREDDITS):
            f = reddit_dir / f"{sub}.jsonl"
            n_lines = sum(1 for _ in open(str(f), 'r'))
            print(f"    {sub}: {n_lines} comments")
        return

    # Accumulators per subreddit
    sub_files = {}
    sub_counts = {s: 0 for s in TARGET_SUBREDDITS}

    for month_file in REDDIT_MONTHS:
        bz2_path = reddit_dir / month_file
        url = f"{ZENODO_BASE}/{month_file}/content"

        if not bz2_path.exists():
            print(f"\n  Downloading {month_file}...")
            print(f"    URL: {url}")
            try:
                urllib.request.urlretrieve(url, str(bz2_path))
                size_mb = bz2_path.stat().st_size / (1024 * 1024)
                print(f"    Downloaded: {size_mb:.0f} MB")
            except Exception as e:
                print(f"    Failed to download: {e}")
                continue
        else:
            size_mb = bz2_path.stat().st_size / (1024 * 1024)
            print(f"\n  {month_file} already downloaded ({size_mb:.0f} MB)")

        # Process the BZ2 file: extract comments for target subreddits
        print(f"  Filtering by target subreddits...")
        count = 0
        matched = 0
        try:
            with bz2.open(str(bz2_path), 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    count += 1
                    if count % 500000 == 0:
                        print(f"    Processed {count:,} lines, matched {matched:,}...")
                    try:
                        post = json.loads(line)
                        sub = post.get('subreddit', '')
                        if sub in TARGET_SUBREDDITS:
                            body = post.get('body', '')
                            if body and body not in ('[removed]', '[deleted]') and len(body) > 20:
                                if sub not in sub_files:
                                    sub_files[sub] = open(
                                        str(reddit_dir / f"{sub}.jsonl"), 'a',
                                        encoding='utf-8')
                                sub_files[sub].write(line)
                                sub_counts[sub] += 1
                                matched += 1
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
        except Exception as e:
            print(f"    Error processing {month_file}: {e}")

        print(f"  {month_file}: {count:,} total lines, {matched:,} matched")

    # Close file handles
    for fh in sub_files.values():
        fh.close()

    print(f"\nPer-subreddit counts:")
    for sub in sorted(TARGET_SUBREDDITS):
        print(f"  r/{sub}: {sub_counts.get(sub, 0):,} comments")


# ====================================================================
# Experiment 2: EEG Hyperscanning
# ====================================================================

def download_eeg():
    """
    Download a publicly available hyperscanning EEG dataset.

    Uses the Cooperation/Competition EEG dataset from Czeszumski et al.
    Figshare collection: 10.6084/m9.figshare.c.7062272

    Alternative: OpenNeuro ds004013 (Social Brain in Action)
    """
    import urllib.request

    eeg_dir = DATA_DIR / "eeg_collab"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Downloading EEG hyperscanning data...")
    print(f"{'='*60}")

    # First, try to get the Figshare article files
    # The collection page lists individual datasets
    # We'll use the API to discover files

    figshare_api = "https://api.figshare.com/v2"
    collection_id = "7062272"

    try:
        import urllib.request
        req = urllib.request.Request(
            f"{figshare_api}/collections/{collection_id}/articles",
            headers={'Accept': 'application/json'}
        )
        with urllib.request.urlopen(req) as resp:
            articles = json.loads(resp.read().decode())

        print(f"  Found {len(articles)} articles in collection")

        # Download first 5 articles (dyads) for tractability
        max_dyads = 8
        downloaded = 0

        for article in articles[:max_dyads * 2]:
            article_id = article['id']

            # Get article details
            req2 = urllib.request.Request(
                f"{figshare_api}/articles/{article_id}",
                headers={'Accept': 'application/json'}
            )
            with urllib.request.urlopen(req2) as resp2:
                details = json.loads(resp2.read().decode())

            title = details.get('title', f'article_{article_id}')
            files = details.get('files', [])

            if not files:
                continue

            # Create dyad directory
            dyad_dir = eeg_dir / f"dyad_{downloaded:02d}"
            dyad_dir.mkdir(exist_ok=True)

            print(f"\n  Article: {title[:60]}...")
            for file_info in files:
                fname = file_info['name']
                furl = file_info['download_url']
                fsize = file_info.get('size', 0)

                out_path = dyad_dir / fname
                if out_path.exists():
                    print(f"    {fname}: already exists")
                    continue

                # Skip very large files (>500MB)
                if fsize > 500 * 1024 * 1024:
                    print(f"    {fname}: too large ({fsize/1024/1024:.0f}MB), skipping")
                    continue

                print(f"    Downloading {fname} ({fsize/1024/1024:.1f}MB)...")
                try:
                    urllib.request.urlretrieve(furl, str(out_path))
                except Exception as e:
                    print(f"    Failed: {e}")

            downloaded += 1
            if downloaded >= max_dyads:
                break

        print(f"\n  Downloaded {downloaded} dyads to {eeg_dir}")

    except Exception as e:
        print(f"  Figshare download failed: {e}")
        print("  Falling back to synthetic data for Exp 2")


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real-world datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--crypto", action="store_true", help="Download crypto data")
    parser.add_argument("--reddit", action="store_true", help="Download Reddit data")
    parser.add_argument("--eeg", action="store_true", help="Download EEG data")
    args = parser.parse_args()

    if not any([args.all, args.crypto, args.reddit, args.eeg]):
        args.all = True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")

    if args.all or args.crypto:
        download_crypto()

    if args.all or args.reddit:
        download_reddit()

    if args.all or args.eeg:
        download_eeg()

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")
