"""
Experiment 2: EEG/Hyperscanning Dyad
======================================
Pipeline for analyzing inter-brain coupling via TDA and curvature.

Data: OpenNeuro hyperscanning EEG datasets (BIDS format)

Tests:
  H1: Neural attractor reconstruction via Takens + TDA
  H3: Forman-Ricci curvature predicts cooperation/rupture
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import warnings


# ====================================================================
# EEG Preprocessing
# ====================================================================

@dataclass
class EEGConfig:
    """Configuration for EEG preprocessing."""
    sfreq: float = 500.0           # sampling frequency (Hz)
    lowcut: float = 1.0            # bandpass low (Hz)
    highcut: float = 45.0          # bandpass high (Hz)
    filter_order: int = 4
    n_ica_components: int = 20
    epoch_tmin: float = -0.5       # epoch start (s)
    epoch_tmax: float = 2.0        # epoch end (s)
    baseline_tmin: float = -0.5
    baseline_tmax: float = 0.0
    freq_bands: dict = None

    def __post_init__(self):
        if self.freq_bands is None:
            self.freq_bands = {
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45),
            }


def bandpass_filter(data: np.ndarray, sfreq: float,
                    lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    from scipy.signal import butter, filtfilt
    nyq = sfreq / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_psd(data: np.ndarray, sfreq: float,
                window_sec: float = 2.0, overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density via Welch's method.

    Parameters
    ----------
    data : (n_channels, n_times) array
    sfreq : float
    window_sec : window length in seconds
    overlap : fractional overlap

    Returns
    -------
    freqs : (n_freqs,) array
    psd : (n_channels, n_freqs) array
    """
    from scipy.signal import welch
    nperseg = int(window_sec * sfreq)
    noverlap = int(nperseg * overlap)
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
    return freqs, psd


def band_power(psd: np.ndarray, freqs: np.ndarray,
               band: tuple[float, float]) -> np.ndarray:
    """Extract band power from PSD."""
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[:, idx], axis=1)


# ====================================================================
# Phase Locking Value (PLV) for inter-brain connectivity
# ====================================================================

def compute_plv(signal1: np.ndarray, signal2: np.ndarray,
                sfreq: float, band: tuple[float, float],
                window_samples: int = 500) -> np.ndarray:
    """
    Compute time-resolved Phase Locking Value between two signals.

    Parameters
    ----------
    signal1, signal2 : (n_times,) arrays
    sfreq : sampling rate
    band : (low, high) frequency band for filtering
    window_samples : sliding window size

    Returns
    -------
    plv_ts : (n_windows,) array of PLV values over time
    """
    from scipy.signal import hilbert, butter, filtfilt

    # Bandpass filter
    nyq = sfreq / 2.0
    b, a = butter(4, [band[0] / nyq, band[1] / nyq], btype='band')
    s1_filt = filtfilt(b, a, signal1)
    s2_filt = filtfilt(b, a, signal2)

    # Analytic signal
    phase1 = np.angle(hilbert(s1_filt))
    phase2 = np.angle(hilbert(s2_filt))

    # Phase difference
    phase_diff = phase1 - phase2

    # Sliding window PLV
    n_windows = len(phase_diff) - window_samples + 1
    if n_windows <= 0:
        return np.array([np.abs(np.mean(np.exp(1j * phase_diff)))])

    plv_ts = np.zeros(n_windows)
    for i in range(n_windows):
        window = phase_diff[i:i + window_samples]
        plv_ts[i] = np.abs(np.mean(np.exp(1j * window)))

    return plv_ts


def compute_plv_matrix(data1: np.ndarray, data2: np.ndarray,
                       sfreq: float, band: tuple[float, float],
                       window_samples: int = 500) -> np.ndarray:
    """
    Compute PLV matrix between all channel pairs across two participants.

    Parameters
    ----------
    data1 : (n_channels, n_times) — participant 1
    data2 : (n_channels, n_times) — participant 2

    Returns
    -------
    plv_matrix : (n_ch1, n_ch2, n_windows) array
    """
    n_ch1, n_times = data1.shape
    n_ch2 = data2.shape[0]
    n_windows = max(1, n_times - window_samples + 1)

    plv_matrix = np.zeros((n_ch1, n_ch2, n_windows))
    for i in range(n_ch1):
        for j in range(n_ch2):
            plv_matrix[i, j, :] = compute_plv(
                data1[i], data2[j], sfreq, band, window_samples
            )[:n_windows]

    return plv_matrix


# ====================================================================
# Forman-Ricci Curvature on interaction graphs
# ====================================================================

def build_interaction_graph(plv_matrix: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Build interaction graph from PLV matrix at a single time point.

    Parameters
    ----------
    plv_matrix : (n_ch1, n_ch2) array of PLV values
    threshold : PLV threshold for edge inclusion

    Returns
    -------
    adj : (n_nodes, n_nodes) weighted adjacency matrix
          n_nodes = n_ch1 + n_ch2
    """
    n_ch1, n_ch2 = plv_matrix.shape
    n_nodes = n_ch1 + n_ch2
    adj = np.zeros((n_nodes, n_nodes))

    for i in range(n_ch1):
        for j in range(n_ch2):
            if plv_matrix[i, j] >= threshold:
                adj[i, n_ch1 + j] = plv_matrix[i, j]
                adj[n_ch1 + j, i] = plv_matrix[i, j]

    return adj


def forman_ricci_curvature(adj: np.ndarray) -> np.ndarray:
    """
    Compute Forman-Ricci curvature for all edges in a weighted graph.

    For an edge e = (v1, v2) with weight w(e):
    Ric_F(e) = w(e) * (w(v1)/w(e) + w(v2)/w(e)
               - sum_{e' ~ v1, e'!=e} w(v1)/sqrt(w(e)*w(e'))
               - sum_{e' ~ v2, e'!=e} w(v2)/sqrt(w(e)*w(e')))

    where w(v) = sum of weights of edges incident to v.

    Parameters
    ----------
    adj : (n, n) weighted adjacency matrix

    Returns
    -------
    curvatures : dict mapping (i, j) -> Ric_F(i, j) for all edges
    mean_curvature : float — mean curvature over all edges
    """
    n = adj.shape[0]

    # Vertex weights = sum of incident edge weights
    vertex_weights = np.sum(adj, axis=1)

    curvatures = {}
    for i in range(n):
        for j in range(i + 1, n):
            w_e = adj[i, j]
            if w_e <= 0:
                continue

            w_v1 = vertex_weights[i]
            w_v2 = vertex_weights[j]

            # Parallel edges at v1 (excluding e)
            sum_v1 = 0.0
            for k in range(n):
                if k != j and adj[i, k] > 0:
                    sum_v1 += w_v1 / np.sqrt(w_e * adj[i, k])

            # Parallel edges at v2 (excluding e)
            sum_v2 = 0.0
            for k in range(n):
                if k != i and adj[j, k] > 0:
                    sum_v2 += w_v2 / np.sqrt(w_e * adj[j, k])

            ric = w_e * (w_v1 / w_e + w_v2 / w_e - sum_v1 - sum_v2)
            curvatures[(i, j)] = ric

    if curvatures:
        mean_curv = np.mean(list(curvatures.values()))
    else:
        mean_curv = 0.0

    return curvatures, mean_curv


def curvature_time_series(plv_matrix_3d: np.ndarray,
                          threshold: float = 0.3) -> np.ndarray:
    """
    Compute Forman-Ricci curvature time series from time-resolved PLV.

    Parameters
    ----------
    plv_matrix_3d : (n_ch1, n_ch2, n_windows) array

    Returns
    -------
    curv_ts : (n_windows,) array of mean curvature over time
    """
    n_windows = plv_matrix_3d.shape[2]
    curv_ts = np.zeros(n_windows)

    for t in range(n_windows):
        adj = build_interaction_graph(plv_matrix_3d[:, :, t], threshold)
        _, mean_curv = forman_ricci_curvature(adj)
        curv_ts[t] = mean_curv

    return curv_ts


# ====================================================================
# Takens + TDA on EEG features
# ====================================================================

def eeg_takens_tda(feature_ts: np.ndarray, tau: int = 5, d_e: int = 8,
                   max_dim: int = 1, n_subsample: int = 300):
    """
    Apply Takens embedding + persistent homology to EEG feature time series.

    Parameters
    ----------
    feature_ts : (T, d) array of time-frequency features
    tau, d_e : Takens embedding parameters
    max_dim : maximum homology dimension
    n_subsample : subsample size for TDA

    Returns
    -------
    diagram : persistence diagram
    embedded : embedded point cloud
    """
    from exp1_synthetic import takens_embedding, compute_persistence

    embedded = takens_embedding(feature_ts, tau=tau, d_e=d_e)
    diagram = compute_persistence(embedded, max_dim=max_dim, n_subsample=n_subsample)

    return diagram, embedded


# ====================================================================
# Cross-correlation analysis
# ====================================================================

def time_lagged_correlation(x: np.ndarray, y: np.ndarray,
                            max_lag: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time-lagged cross-correlation between two time series.

    Parameters
    ----------
    x, y : (T,) arrays
    max_lag : maximum lag in samples

    Returns
    -------
    lags : (2*max_lag + 1,) array of lag values
    corrs : (2*max_lag + 1,) array of correlation values
    """
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)
    n = len(x)

    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags))

    for idx, lag in enumerate(lags):
        if lag >= 0:
            corrs[idx] = np.mean(x[:n - lag] * y[lag:])
        else:
            corrs[idx] = np.mean(x[-lag:] * y[:n + lag])

    return lags, corrs


# ====================================================================
# Synthetic EEG demo (for testing without real data)
# ====================================================================

def generate_synthetic_eeg(n_channels: int = 16, n_times: int = 50000,
                           sfreq: float = 500.0,
                           coupling_strength: float = 0.3,
                           seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic dual-EEG data with time-varying coupling.

    Models two participants with Kuramoto-like oscillators that
    periodically synchronize (cooperation) and desynchronize (rupture).

    Returns
    -------
    data1 : (n_channels, n_times) — participant 1
    data2 : (n_channels, n_times) — participant 2
    coupling_signal : (n_times,) — ground truth coupling strength
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / sfreq

    # Natural frequencies for each channel (Hz)
    freqs1 = rng.uniform(8, 12, size=n_channels)  # alpha band
    freqs2 = rng.uniform(8, 12, size=n_channels)

    # Time-varying coupling: oscillates between high and low
    t = np.arange(n_times) / sfreq
    # Coupling modulation: cycles of ~10s synchrony, ~10s independence
    coupling_signal = coupling_strength * (0.5 + 0.5 * np.sin(2 * np.pi * t / 20.0))

    # Phases
    phases1 = np.zeros((n_channels, n_times))
    phases2 = np.zeros((n_channels, n_times))
    phases1[:, 0] = rng.uniform(0, 2 * np.pi, n_channels)
    phases2[:, 0] = rng.uniform(0, 2 * np.pi, n_channels)

    for tt in range(1, n_times):
        kappa = coupling_signal[tt]

        for ch in range(n_channels):
            # Kuramoto update with coupling to matched channel
            phase_diff = phases2[ch, tt - 1] - phases1[ch, tt - 1]
            phases1[ch, tt] = (
                phases1[ch, tt - 1]
                + 2 * np.pi * freqs1[ch] * dt
                + kappa * np.sin(phase_diff) * dt
                + rng.normal(0, 0.1) * np.sqrt(dt)
            )

            phase_diff2 = phases1[ch, tt - 1] - phases2[ch, tt - 1]
            phases2[ch, tt] = (
                phases2[ch, tt - 1]
                + 2 * np.pi * freqs2[ch] * dt
                + kappa * np.sin(phase_diff2) * dt
                + rng.normal(0, 0.1) * np.sqrt(dt)
            )

    # Convert phases to signals (add harmonics + noise)
    data1 = np.sin(phases1) + 0.3 * np.sin(2 * phases1) + rng.normal(0, 0.5, phases1.shape)
    data2 = np.sin(phases2) + 0.3 * np.sin(2 * phases2) + rng.normal(0, 0.5, phases2.shape)

    return data1, data2, coupling_signal


# ====================================================================
# Main experiment runner
# ====================================================================

def run_experiment_synthetic(seed: int = 42):
    """Run Experiment 2 on synthetic EEG data."""
    print("Generating synthetic dual-EEG data...")
    data1, data2, coupling = generate_synthetic_eeg(
        n_channels=16, n_times=50000, sfreq=500.0,
        coupling_strength=0.6, seed=seed,  # stronger coupling for clearer signal
    )
    sfreq = 500.0
    config = EEGConfig(sfreq=sfreq)
    print(f"  Data shape: {data1.shape}, {data2.shape}")

    # Bandpass filter
    print("Filtering...")
    data1_f = bandpass_filter(data1, sfreq, config.lowcut, config.highcut)
    data2_f = bandpass_filter(data2, sfreq, config.lowcut, config.highcut)

    # Compute PLV in alpha band with 2s non-overlapping windows
    # This gives ~50 PLV snapshots (100s / 2s), matching coupling timescale
    print("Computing PLV (alpha band, 2s windows)...")
    alpha_band = config.freq_bands['alpha']
    window_samples = int(2.0 * sfreq)  # 2s window for stable PLV

    # Use subset of channels for speed
    n_ch_sub = 8

    # Non-overlapping windowed PLV for cleaner temporal resolution
    n_times = data1_f.shape[1]
    n_blocks = n_times // window_samples
    print(f"  Computing PLV for {n_blocks} non-overlapping 2s blocks...")

    plv_blocks = np.zeros((n_ch_sub, n_ch_sub, n_blocks))
    from scipy.signal import hilbert, butter, filtfilt
    nyq = sfreq / 2.0
    b, a = butter(4, [alpha_band[0] / nyq, alpha_band[1] / nyq], btype='band')

    # Filter once, then window the phases
    phases1 = np.angle(hilbert(filtfilt(b, a, data1_f[:n_ch_sub], axis=-1), axis=-1))
    phases2 = np.angle(hilbert(filtfilt(b, a, data2_f[:n_ch_sub], axis=-1), axis=-1))

    for blk in range(n_blocks):
        s = blk * window_samples
        e = s + window_samples
        for i in range(n_ch_sub):
            for j in range(n_ch_sub):
                phase_diff = phases1[i, s:e] - phases2[j, s:e]
                plv_blocks[i, j, blk] = np.abs(np.mean(np.exp(1j * phase_diff)))

    print(f"  PLV blocks shape: {plv_blocks.shape}")

    # Forman-Ricci curvature time series (one per 2s block)
    print("Computing Forman-Ricci curvature...")
    curv_ts = curvature_time_series(plv_blocks, threshold=0.3)
    print(f"  Curvature time series length: {len(curv_ts)}")

    # Downsample coupling signal: one value per 2s block (block center)
    coupling_ds = np.array([
        np.mean(coupling[blk * window_samples:(blk + 1) * window_samples])
        for blk in range(n_blocks)
    ])

    # Smooth curvature with 3-block (~6s) moving average
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    curv_smooth = np.convolve(curv_ts, kernel, mode='same')

    # Time-lagged correlation between smoothed curvature and coupling
    print("Computing time-lagged correlation...")
    max_lag_blocks = min(5, len(curv_smooth) // 4)  # ~10s max lag
    lags, corrs = time_lagged_correlation(curv_smooth, coupling_ds, max_lag=max_lag_blocks)

    peak_lag = lags[np.argmax(np.abs(corrs))]
    peak_corr = corrs[np.argmax(np.abs(corrs))]
    print(f"  Peak correlation: r={peak_corr:.4f} at lag={peak_lag}")

    # H1: TDA on EEG features
    print("\n--- H1: Neural attractor reconstruction ---")
    # Extract band power features over time
    from scipy.signal import welch
    feature_list = []
    win = int(2.0 * sfreq)
    hop = int(0.5 * sfreq)
    n_frames = max(1, (data1_f.shape[1] - win) // hop)

    for t_idx in range(min(n_frames, 200)):
        start = t_idx * hop
        end = start + win
        if end > data1_f.shape[1]:
            break
        segment = data1_f[:, start:end]
        freqs_psd, psd = compute_psd(segment, sfreq, window_sec=1.0)

        features = []
        for band_name, band_range in config.freq_bands.items():
            bp = band_power(psd, freqs_psd, band_range)
            features.append(bp)
        feature_list.append(np.concatenate(features))

    features_arr = np.array(feature_list)  # (n_frames, n_channels * n_bands)
    print(f"  Feature matrix shape: {features_arr.shape}")

    if features_arr.shape[0] > 30:
        from exp1_synthetic import takens_embedding, compute_persistence
        embedded = takens_embedding(features_arr, tau=3, d_e=5)
        dgm = compute_persistence(embedded, max_dim=1, n_subsample=200)
        print(f"  Persistence diagram: {len(dgm)} features")
    else:
        print("  Insufficient data for TDA")

    results = {
        'peak_correlation': peak_corr,
        'peak_lag': peak_lag,
        'mean_curvature': np.mean(curv_ts),
        'curvature_std': np.std(curv_ts),
        'plv_mean': np.mean(plv_blocks),
    }

    return results, curv_ts, coupling_ds


if __name__ == '__main__':
    results, curv, coupling = run_experiment_synthetic(seed=42)
    print("\n=== Experiment 2 Complete ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
