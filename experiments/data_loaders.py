"""
Data loaders for real-world datasets used in the Belief Geodesics Framework.
===========================================================================

Centralizes all real data loading: crypto price data (Exp 1),
hyperscanning EEG/fNIRS (Exp 2), and Reddit Politosphere (Exp 3).
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import warnings
import json


# ====================================================================
# Experiment 1: Crypto asset data
# ====================================================================

@dataclass
class CryptoAssetData:
    """Container for a single crypto asset's price data."""
    name: str
    prices: np.ndarray          # daily close prices
    returns: np.ndarray         # log returns
    timestamps: np.ndarray      # datetime strings or unix
    volume: np.ndarray          # daily volume
    volatility_regime: str = '' # 'stable', 'large_cap', 'mid_cap'


def load_crypto_csv(filepath: str) -> CryptoAssetData:
    """
    Load a crypto asset from CryptoDataDownload CSV format.

    Expected columns: Date, Open, High, Low, Close, Volume
    (or unix, open, high, low, close, Volume depending on source).
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    # Normalize column names (CryptoDataDownload uses varying conventions)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ('date', 'timestamp', 'unix', 'time'):
            col_map['date'] = col
        elif cl == 'close':
            col_map['close'] = col
        elif cl == 'high':
            col_map['high'] = col
        elif cl == 'low':
            col_map['low'] = col
        elif cl in ('volume', 'volume usd', 'vol'):
            col_map['volume'] = col

    if 'close' not in col_map:
        raise ValueError(f"No 'close' column found in {filepath}. Columns: {list(df.columns)}")

    prices = df[col_map['close']].values.astype(float)
    timestamps = df[col_map.get('date', df.columns[0])].values

    # Handle NaNs
    valid = ~np.isnan(prices)
    prices = prices[valid]
    timestamps = timestamps[valid]

    volume = np.zeros(len(prices))
    if 'volume' in col_map:
        vol_raw = df[col_map['volume']].values.astype(float)[valid]
        volume = np.nan_to_num(vol_raw, 0.0)

    # Compute log returns
    returns = np.diff(np.log(np.maximum(prices, 1e-10)))

    # Trim prices/timestamps/volume to match returns length
    prices = prices[1:]
    timestamps = timestamps[1:]
    volume = volume[1:]

    name = Path(filepath).stem
    return CryptoAssetData(
        name=name,
        prices=prices,
        returns=returns,
        timestamps=timestamps,
        volume=volume,
    )


def load_crypto_dir(data_dir: str) -> list[CryptoAssetData]:
    """
    Load all crypto CSVs from a directory and classify by volatility regime.

    Regime classification by annualized volatility:
    - stable: vol < 0.3 (stablecoins, BTC in calm periods)
    - large_cap: 0.3 <= vol < 1.0
    - mid_cap: vol >= 1.0
    """
    data_path = Path(data_dir)
    assets = []

    for f in sorted(data_path.glob('*.csv')):
        try:
            asset = load_crypto_csv(str(f))
            if len(asset.returns) < 30:
                continue

            # Classify by annualized volatility
            ann_vol = np.std(asset.returns) * np.sqrt(365)
            if ann_vol < 0.3:
                asset.volatility_regime = 'stable'
            elif ann_vol < 1.0:
                asset.volatility_regime = 'large_cap'
            else:
                asset.volatility_regime = 'mid_cap'

            assets.append(asset)
        except Exception as e:
            warnings.warn(f"Failed to load {f.name}: {e}")

    return assets


def crypto_feature_vector(asset: CryptoAssetData, window: int = 5) -> np.ndarray:
    """
    Compute daily feature vectors for a crypto asset.

    Features per day (d=4):
    [log_return, realized_vol_5d, volume_ratio, high_low_range_proxy]

    Returns (T-window, 4) array.
    """
    T = len(asset.returns)
    if T < window + 1:
        return np.empty((0, 4))

    features = np.zeros((T - window, 4))

    for t in range(window, T):
        # Log return
        features[t - window, 0] = asset.returns[t]

        # Realized volatility over window
        features[t - window, 1] = np.std(asset.returns[t - window:t])

        # Volume ratio (current / rolling mean)
        mean_vol = np.mean(asset.volume[t - window:t])
        if mean_vol > 0:
            features[t - window, 2] = asset.volume[t] / mean_vol
        else:
            features[t - window, 2] = 1.0

        # Price range proxy (return dispersion in window)
        features[t - window, 3] = np.max(asset.returns[t - window:t]) - np.min(asset.returns[t - window:t])

    return features


# ====================================================================
# Experiment 2: Hyperscanning data
# ====================================================================

@dataclass
class HyperscanningData:
    """Container for a hyperscanning dyad's data."""
    participant1: np.ndarray     # (n_channels, n_times)
    participant2: np.ndarray     # (n_channels, n_times)
    sfreq: float                 # sampling frequency (Hz)
    condition_labels: Optional[np.ndarray] = None  # per-timepoint or per-epoch labels
    modality: str = 'eeg'        # 'eeg' or 'fnirs'
    dyad_id: str = ''
    channel_names: Optional[list] = None


def load_collaboration_eeg(data_dir: str, dyad_id: int) -> Optional[HyperscanningData]:
    """
    Load collaboration/competition EEG dataset from Figshare.

    Dataset: 4-channel EEG recorded during cooperative and competitive tasks.
    Figshare collection: 10.6084/m9.figshare.c.7062272

    Expected directory structure:
        data_dir/
            dyad_XX/
                participant1.csv  (or .npy or .edf)
                participant2.csv
                conditions.csv

    Falls back to MNE-compatible formats (.edf, .bdf, .set).
    """
    dyad_dir = Path(data_dir) / f"dyad_{dyad_id:02d}"
    if not dyad_dir.exists():
        dyad_dir = Path(data_dir) / f"dyad{dyad_id}"
    if not dyad_dir.exists():
        return None

    # Try numpy format first (preprocessed)
    p1_npy = dyad_dir / 'participant1.npy'
    p2_npy = dyad_dir / 'participant2.npy'
    if p1_npy.exists() and p2_npy.exists():
        p1 = np.load(str(p1_npy))
        p2 = np.load(str(p2_npy))
        conditions = None
        cond_file = dyad_dir / 'conditions.npy'
        if cond_file.exists():
            conditions = np.load(str(cond_file))
        return HyperscanningData(
            participant1=p1, participant2=p2,
            sfreq=500.0, condition_labels=conditions,
            modality='eeg', dyad_id=f"collab_{dyad_id}",
        )

    # Try MNE for EDF/BDF files
    try:
        import mne
        edf_files = list(dyad_dir.glob('*.edf')) + list(dyad_dir.glob('*.bdf'))
        if len(edf_files) >= 2:
            raw1 = mne.io.read_raw(str(edf_files[0]), preload=True, verbose=False)
            raw2 = mne.io.read_raw(str(edf_files[1]), preload=True, verbose=False)
            return HyperscanningData(
                participant1=raw1.get_data(),
                participant2=raw2.get_data(),
                sfreq=raw1.info['sfreq'],
                modality='eeg',
                dyad_id=f"collab_{dyad_id}",
                channel_names=raw1.ch_names,
            )
    except ImportError:
        warnings.warn("MNE not available for EDF loading")

    # Try CSV format
    csv_files = sorted(dyad_dir.glob('*.csv'))
    if len(csv_files) >= 2:
        import pandas as pd
        p1 = pd.read_csv(str(csv_files[0])).values.T  # assume channels x times
        p2 = pd.read_csv(str(csv_files[1])).values.T
        conditions = None
        if len(csv_files) >= 3:
            conditions = pd.read_csv(str(csv_files[2])).values.flatten()
        return HyperscanningData(
            participant1=p1.astype(float),
            participant2=p2.astype(float),
            sfreq=500.0,
            condition_labels=conditions,
            modality='eeg',
            dyad_id=f"collab_{dyad_id}",
        )

    return None


def load_parent_child_fnirs(data_dir: str, dyad_id: int) -> Optional[HyperscanningData]:
    """
    Load parent-child fNIRS dataset.

    Dataset: Nature Scientific Data 10.1038/s41597-022-01751-2
    62 parent-child dyads, fNIRS hyperscanning during free play vs video.

    Expected: BIDS-like structure or preprocessed .npy files.
    """
    dyad_dir = Path(data_dir) / f"sub-{dyad_id:02d}"
    if not dyad_dir.exists():
        dyad_dir = Path(data_dir) / f"dyad_{dyad_id:02d}"
    if not dyad_dir.exists():
        return None

    # Preprocessed numpy format
    p1_npy = dyad_dir / 'parent.npy'
    p2_npy = dyad_dir / 'child.npy'
    if p1_npy.exists() and p2_npy.exists():
        p1 = np.load(str(p1_npy))
        p2 = np.load(str(p2_npy))
        conditions = None
        cond_file = dyad_dir / 'conditions.npy'
        if cond_file.exists():
            conditions = np.load(str(cond_file))
        return HyperscanningData(
            participant1=p1, participant2=p2,
            sfreq=7.8125,  # typical fNIRS sampling rate
            condition_labels=conditions,
            modality='fnirs', dyad_id=f"parent_child_{dyad_id}",
        )

    # Try MNE-NIRS for SNIRF files
    try:
        import mne
        snirf_files = list(dyad_dir.glob('*.snirf'))
        if len(snirf_files) >= 2:
            raw1 = mne.io.read_raw_snirf(str(snirf_files[0]), preload=True, verbose=False)
            raw2 = mne.io.read_raw_snirf(str(snirf_files[1]), preload=True, verbose=False)
            # Convert to HbO/HbR
            raw1 = mne.preprocessing.nirs.optical_density(raw1)
            raw1 = mne.preprocessing.nirs.beer_lambert_law(raw1)
            raw2 = mne.preprocessing.nirs.optical_density(raw2)
            raw2 = mne.preprocessing.nirs.beer_lambert_law(raw2)
            # Select HbO channels only
            raw1_hbo = raw1.copy().pick(picks='hbo')
            raw2_hbo = raw2.copy().pick(picks='hbo')
            return HyperscanningData(
                participant1=raw1_hbo.get_data(),
                participant2=raw2_hbo.get_data(),
                sfreq=raw1.info['sfreq'],
                modality='fnirs',
                dyad_id=f"parent_child_{dyad_id}",
                channel_names=raw1_hbo.ch_names,
            )
    except (ImportError, Exception) as e:
        warnings.warn(f"MNE fNIRS loading failed for dyad {dyad_id}: {e}")

    return None


def load_social_touch_fnirs(data_dir: str, dyad_id: int) -> Optional[HyperscanningData]:
    """
    Load social touch fNIRS dataset.

    Dataset: OSF 10.17605/OSF.IO/SM7YT
    47 dyads, friend vs stranger, touch/no-touch conditions.

    Expected: preprocessed .npy or SNIRF files.
    """
    dyad_dir = Path(data_dir) / f"dyad_{dyad_id:02d}"
    if not dyad_dir.exists():
        return None

    # Preprocessed numpy format
    p1_npy = dyad_dir / 'person1.npy'
    p2_npy = dyad_dir / 'person2.npy'
    if p1_npy.exists() and p2_npy.exists():
        p1 = np.load(str(p1_npy))
        p2 = np.load(str(p2_npy))
        conditions = None
        cond_file = dyad_dir / 'conditions.npy'
        if cond_file.exists():
            conditions = np.load(str(cond_file))
        return HyperscanningData(
            participant1=p1, participant2=p2,
            sfreq=7.8125,
            condition_labels=conditions,
            modality='fnirs', dyad_id=f"touch_{dyad_id}",
        )

    # Try MNE-NIRS for SNIRF files (same pattern as parent-child)
    try:
        import mne
        snirf_files = list(dyad_dir.glob('*.snirf'))
        if len(snirf_files) >= 2:
            raw1 = mne.io.read_raw_snirf(str(snirf_files[0]), preload=True, verbose=False)
            raw2 = mne.io.read_raw_snirf(str(snirf_files[1]), preload=True, verbose=False)
            raw1 = mne.preprocessing.nirs.optical_density(raw1)
            raw1 = mne.preprocessing.nirs.beer_lambert_law(raw1)
            raw2 = mne.preprocessing.nirs.optical_density(raw2)
            raw2 = mne.preprocessing.nirs.beer_lambert_law(raw2)
            raw1_hbo = raw1.copy().pick(picks='hbo')
            raw2_hbo = raw2.copy().pick(picks='hbo')
            return HyperscanningData(
                participant1=raw1_hbo.get_data(),
                participant2=raw2_hbo.get_data(),
                sfreq=raw1.info['sfreq'],
                modality='fnirs',
                dyad_id=f"touch_{dyad_id}",
                channel_names=raw1_hbo.ch_names,
            )
    except (ImportError, Exception) as e:
        warnings.warn(f"MNE fNIRS loading failed for dyad {dyad_id}: {e}")

    return None


# ====================================================================
# Experiment 3: Reddit Politosphere
# ====================================================================

# Subreddit ideological classification
SUBREDDIT_LABELS = {
    # Echo-left
    'socialism': 'echo_left',
    'LateStageCapitalism': 'echo_left',
    'ChapoTrapHouse': 'echo_left',
    # Echo-right
    'Conservative': 'echo_right',
    'The_Donald': 'echo_right',
    'Republican': 'echo_right',
    # Diverse
    'PoliticalDiscussion': 'diverse',
    'NeutralPolitics': 'diverse',
    'moderatepolitics': 'diverse',
    # Polarized (bimodal left/right)
    'politics': 'polarized',
}


def load_reddit_politosphere(data_dir: str, subreddit: str,
                              max_posts: int = 50000) -> 'SubredditData':
    """
    Load Reddit Politosphere data (Zenodo: 10.5281/zenodo.5851729).

    The Politosphere dataset stores comments in BZ2-compressed JSONL files,
    one per subreddit. Each line is a JSON object with fields including
    'body', 'author', 'created_utc', 'subreddit', 'score'.

    Parameters
    ----------
    data_dir : str
        Path to the reddit data directory.
    subreddit : str
        Name of the subreddit to load.
    max_posts : int
        Maximum number of posts to load.

    Returns
    -------
    SubredditData from exp3_social_media module.
    """
    from exp3_social_media import SubredditData

    data_path = Path(data_dir)
    texts = []
    timestamps = []

    # Try BZ2 compressed format first (native Politosphere format)
    bz2_file = data_path / f"{subreddit}.bz2"
    if not bz2_file.exists():
        bz2_file = data_path / f"RC_{subreddit}.bz2"
    if not bz2_file.exists():
        bz2_file = data_path / f"{subreddit}.jsonl.bz2"

    if bz2_file.exists():
        import bz2
        with bz2.open(str(bz2_file), 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if len(texts) >= max_posts:
                    break
                try:
                    post = json.loads(line)
                    # Politosphere uses 'body' for comment text
                    text = post.get('body', '') or post.get('selftext', '') or post.get('title', '')
                    if text in ('[removed]', '[deleted]', ''):
                        continue
                    if len(text) > 20:
                        texts.append(text[:2000])  # truncate very long posts
                        timestamps.append(post.get('created_utc', 0))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    else:
        # Fall back to uncompressed JSONL
        jsonl_file = data_path / f"{subreddit}.jsonl"
        if not jsonl_file.exists():
            jsonl_file = data_path / f"{subreddit}.json"
        if jsonl_file.exists():
            with open(str(jsonl_file), 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if len(texts) >= max_posts:
                        break
                    try:
                        post = json.loads(line)
                        text = post.get('body', '') or post.get('selftext', '') or post.get('title', '')
                        if text in ('[removed]', '[deleted]', ''):
                            continue
                        if len(text) > 20:
                            texts.append(text[:2000])
                            timestamps.append(post.get('created_utc', 0))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

    if not texts:
        warnings.warn(f"No data found for r/{subreddit} in {data_dir}")

    label = SUBREDDIT_LABELS.get(subreddit, 'unknown')

    return SubredditData(
        name=f"{subreddit}",
        texts=texts,
        timestamps=np.array(timestamps, dtype=float) if timestamps else None,
    )


def load_precomputed_embeddings(data_dir: str, subreddit: str) -> Optional[np.ndarray]:
    """
    Load precomputed SBERT embeddings for a subreddit.

    Returns (n_posts, 768) array or None if not found.
    """
    emb_dir = Path(data_dir) / 'embeddings'
    emb_file = emb_dir / f"{subreddit}_embeddings.npy"
    if emb_file.exists():
        return np.load(str(emb_file))
    return None
