"""
Plotting utilities for Belief Geodesics Framework paper.
=========================================================

Generates publication-quality figures for all three experiments.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'R': '#d62728',   # red — rigid
    'F': '#2ca02c',   # green — flexible
    'M': '#1f77b4',   # blue — mixed
    'echo': '#d62728',
    'diverse': '#2ca02c',
    'polarized': '#ff7f0e',
    'curvature': '#9467bd',
    'coupling': '#8c564b',
}

FIG_DIR = Path(__file__).parent.parent / 'figures'


def ensure_fig_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ====================================================================
# Experiment 1 Figures
# ====================================================================

def fig_belief_trajectories(trajectories: np.ndarray, group_indices: dict,
                            filename: str = 'fig_exp1_trajectories.pdf'):
    """
    Plot belief trajectories for each group (2D projection).
    """
    ensure_fig_dir()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, (name, idx) in zip(axes, group_indices.items()):
        # Project to first two dimensions
        for i in idx[:30]:
            traj = trajectories[:, i, :2]
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.15, linewidth=0.5,
                    color=COLORS.get(name, 'gray'))
        # Group centroid
        centroid = np.mean(trajectories[:, idx, :2], axis=1)
        ax.plot(centroid[:, 0], centroid[:, 1], color='black',
                linewidth=1.5, label='Centroid')
        ax.set_title(f'Group {name}')
        ax.set_xlabel(r'$\phi_1$')
        if ax == axes[0]:
            ax.set_ylabel(r'$\phi_2$')

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


def fig_persistence_comparison(group_diagrams: dict,
                                filename: str = 'fig_exp1_persistence.pdf'):
    """
    Plot persistence diagrams for representative agents from each group.
    """
    ensure_fig_dir()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for ax, (name, diagrams) in zip(axes, group_diagrams.items()):
        if len(diagrams) == 0 or len(diagrams[0]) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'Group {name}')
            continue

        # Plot first diagram
        dgm = diagrams[0]
        if len(dgm) > 0:
            h0 = dgm[dgm[:, 2] == 0]
            h1 = dgm[dgm[:, 2] == 1]

            if len(h0) > 0:
                ax.scatter(h0[:, 0], h0[:, 1], c=COLORS.get(name, 'gray'),
                           marker='o', s=20, label=r'$H_0$', alpha=0.7)
            if len(h1) > 0:
                ax.scatter(h1[:, 0], h1[:, 1], c=COLORS.get(name, 'gray'),
                           marker='^', s=20, label=r'$H_1$', alpha=0.7)

            # Diagonal
            max_val = np.max(dgm[:, :2]) if len(dgm) > 0 else 1
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=0.5)

        ax.set_title(f'Group {name}')
        ax.set_xlabel('Birth')
        if ax == axes[0]:
            ax.set_ylabel('Death')
        ax.legend(loc='lower right')

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


def fig_rds_vs_kl(rds_dists: dict, kl_dists: dict,
                   filename: str = 'fig_exp1_rds_vs_kl.pdf'):
    """
    Bar chart comparing RDS distance and KL divergence between groups.
    """
    ensure_fig_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    keys = list(rds_dists.keys())
    x = np.arange(len(keys))
    width = 0.35

    rds_vals = [rds_dists[k] for k in keys]
    kl_vals = [kl_dists[k] for k in keys]

    ax1.bar(x, rds_vals, width, color='#1f77b4', label='RDS distance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys)
    ax1.set_ylabel('Distance')
    ax1.set_title('RDS Misalignment Distance')

    ax2.bar(x, kl_vals, width, color='#ff7f0e', label='KL divergence')
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys)
    ax2.set_ylabel('Divergence')
    ax2.set_title('KL Divergence')

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


# ====================================================================
# Experiment 2 Figures
# ====================================================================

def fig_curvature_coupling(curv_ts: np.ndarray, coupling_ts: np.ndarray,
                            sfreq_windows: float = 1.0,
                            filename: str = 'fig_exp2_curvature.pdf'):
    """
    Plot Forman-Ricci curvature vs ground-truth coupling over time.
    """
    ensure_fig_dir()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    t = np.arange(len(curv_ts)) / sfreq_windows

    # Curvature
    ax1.plot(t, curv_ts, color=COLORS['curvature'], linewidth=0.8, alpha=0.8)
    # Smooth
    kernel = np.ones(50) / 50
    if len(curv_ts) > 50:
        curv_smooth = np.convolve(curv_ts, kernel, mode='same')
        ax1.plot(t, curv_smooth, color=COLORS['curvature'], linewidth=2)
    ax1.set_ylabel('Forman-Ricci\nCurvature')
    ax1.set_title('Inter-Brain Curvature Dynamics')

    # Coupling
    ax2.plot(t[:len(coupling_ts)], coupling_ts, color=COLORS['coupling'], linewidth=1.5)
    ax2.set_ylabel('Coupling\nStrength')

    # Overlay (normalized)
    curv_norm = (curv_ts - np.mean(curv_ts)) / (np.std(curv_ts) + 1e-8)
    coup_norm = (coupling_ts - np.mean(coupling_ts)) / (np.std(coupling_ts) + 1e-8)
    ax3.plot(t, curv_norm, color=COLORS['curvature'], linewidth=0.8, alpha=0.5, label='Curvature (z)')
    ax3.plot(t[:len(coup_norm)], coup_norm, color=COLORS['coupling'], linewidth=1.5, label='Coupling (z)')
    ax3.set_ylabel('z-score')
    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


def fig_cross_correlation(lags: np.ndarray, corrs: np.ndarray,
                           filename: str = 'fig_exp2_xcorr.pdf'):
    """
    Plot time-lagged cross-correlation between curvature and cooperation.
    """
    ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(lags, corrs, color='#1f77b4', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    peak_idx = np.argmax(np.abs(corrs))
    ax.scatter(lags[peak_idx], corrs[peak_idx], color='red', s=50, zorder=5)
    ax.annotate(f'Peak: r={corrs[peak_idx]:.3f}\nlag={lags[peak_idx]}',
                xy=(lags[peak_idx], corrs[peak_idx]),
                xytext=(lags[peak_idx] + 5, corrs[peak_idx] - 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9)

    ax.set_xlabel('Lag (windows)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Curvature-Coupling Cross-Correlation')

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


# ====================================================================
# Experiment 3 Figures
# ====================================================================

def fig_distance_matrices(rds_matrix: np.ndarray, kl_matrix: np.ndarray,
                           emb_matrix: np.ndarray, names: list[str],
                           filename: str = 'fig_exp3_distances.pdf'):
    """
    Plot distance matrices (RDS, KL, embedding) as heatmaps.
    """
    ensure_fig_dir()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    matrices = [rds_matrix, kl_matrix, emb_matrix]
    titles = ['RDS Distance', 'KL Divergence', 'Embedding Distance']
    cmaps = ['YlOrRd', 'YlOrRd', 'YlOrRd']

    short_names = [n.replace('community_', '').replace('_', '\n') for n in names]

    for ax, mat, title, cmap in zip(axes, matrices, titles, cmaps):
        # Normalize for display
        if np.max(mat) > 0:
            mat_norm = mat / np.max(mat)
        else:
            mat_norm = mat

        im = ax.imshow(mat_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


def fig_discrimination_comparison(rds_ratio: float, kl_ratio: float,
                                    emb_ratio: float,
                                    filename: str = 'fig_exp3_discrimination.pdf'):
    """
    Bar chart comparing discrimination ratios of different metrics.
    """
    ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    metrics = ['RDS\nDistance', 'KL\nDivergence', 'Embedding\nDistance']
    ratios = [rds_ratio, kl_ratio, emb_ratio]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = ax.bar(metrics, ratios, color=colors, width=0.5, edgecolor='black', linewidth=0.5)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5, label='No discrimination')
    ax.set_ylabel('Between-type / Within-type\nDistance Ratio')
    ax.set_title('Community Type Discrimination')

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


# ====================================================================
# Overview / conceptual figure
# ====================================================================

def fig_framework_overview(filename: str = 'fig_framework_overview.pdf'):
    """
    Conceptual figure showing the 5-phase framework.
    """
    ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(10, 4))

    phases = [
        ('Phase 1', 'Belief\nTopology', '#3498db'),
        ('Phase 1.5', 'Cognitive\nInterpretation', '#9b59b6'),
        ('Phase 2', 'Misalignment\nMetric', '#e74c3c'),
        ('Phase 3', 'Belief\nGeodesics', '#f39c12'),
        ('Phase 4', 'Co-Steering\n& Coarse-Grain', '#2ecc71'),
        ('Phase 4.5', 'Normative\nCriteria', '#1abc9c'),
    ]

    x_positions = np.linspace(0.08, 0.92, len(phases))

    for x, (label, desc, color) in zip(x_positions, phases):
        # Circle
        circle = plt.Circle((x, 0.55), 0.06, facecolor=color, edgecolor='black',
                             linewidth=1.5, transform=ax.transAxes, zorder=3)
        ax.add_patch(circle)
        ax.text(x, 0.55, label.replace('Phase ', ''), ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', transform=ax.transAxes,
                zorder=4)
        ax.text(x, 0.25, f'{label}\n{desc}', ha='center', va='center',
                fontsize=8, transform=ax.transAxes)

    # Arrows
    for i in range(len(phases) - 1):
        ax.annotate('', xy=(x_positions[i + 1] - 0.07, 0.55),
                     xytext=(x_positions[i] + 0.07, 0.55),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                     xycoords='axes fraction', textcoords='axes fraction')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Framework Overview: Geometric Alignment via Belief Dynamics', fontsize=12, pad=20)

    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


# ====================================================================
# Generate all figures
# ====================================================================

def generate_all_figures():
    """Generate all paper figures using experiment results."""
    print("=" * 60)
    print("Generating paper figures")
    print("=" * 60)

    # Framework overview (no data needed)
    fig_framework_overview()

    # Experiment 1
    print("\n--- Experiment 1 Figures ---")
    from exp1_synthetic import run_experiment as run_exp1
    results1, traj1, groups1 = run_exp1(seed=42, n_steps=10000)

    fig_belief_trajectories(traj1, groups1)
    fig_rds_vs_kl(results1['h2_rds'], results1['h2_kl'])

    # Experiment 2
    print("\n--- Experiment 2 Figures ---")
    from exp2_eeg import run_experiment_synthetic as run_exp2
    results2, curv_ts, coupling_ds = run_exp2(seed=42)

    fig_curvature_coupling(curv_ts, coupling_ds)

    # Experiment 3
    print("\n--- Experiment 3 Figures ---")
    from exp3_social_media import run_experiment as run_exp3
    results3, communities = run_exp3(use_synthetic=True, seed=42)

    names = [c.name for c in communities]
    fig_distance_matrices(results3['rds_matrix'], results3['kl_matrix'],
                           results3['emb_matrix'], names)
    fig_discrimination_comparison(results3['rds_discrimination'],
                                   results3['kl_discrimination'],
                                   results3['emb_discrimination'])

    print("\n" + "=" * 60)
    print("All figures generated.")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_figures()
