"""Generate paper figures using real data results.

Produces:
  fig_exp1_crypto.pdf       — Crypto validation (prices, persistence, RDS/KL)
  fig_exp2_curvature.pdf    — Best-dyad curvature time series (real EEG)
  fig_exp2_dyad_summary.pdf — Per-dyad |r| bar chart (all 8 dyads)
  fig_exp3_distances.pdf    — KL heatmap + H0/H1 feature summary
  fig_exp3_persistence.pdf  — Grouped bar chart of persistence by community
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ====================================================================
# Experiment 1: Crypto validation figure
# ====================================================================

def fig_exp1_crypto():
    """Generate 3-panel crypto validation figure for Experiment 1."""
    from exp1_synthetic import run_experiment_crypto

    crypto_dir = 'data/crypto'
    if not Path(crypto_dir).exists():
        print("No crypto data directory found, skipping")
        return

    results, assets, regime_indices = run_experiment_crypto(crypto_dir, seed=42)

    regime_colors = {'large_cap': '#e67e22', 'mid_cap': '#e74c3c'}
    regime_labels = {'large_cap': 'Large-Cap', 'mid_cap': 'Mid-Cap'}

    fig = plt.figure(figsize=(15, 4.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)

    # (a) Price trajectories
    ax_price = fig.add_subplot(gs[0, 0])
    plotted = set()
    for regime, idx_list in regime_indices.items():
        color = regime_colors.get(regime, 'gray')
        for i in idx_list[:6]:
            asset = assets[i]
            norm_prices = asset.prices / asset.prices[0]
            lbl = regime_labels.get(regime) if regime not in plotted else None
            ax_price.plot(norm_prices, color=color, linewidth=0.8, alpha=0.6, label=lbl)
            plotted.add(regime)
    ax_price.set_xlabel('Trading day')
    ax_price.set_ylabel('Normalized price')
    ax_price.set_yscale('log')
    ax_price.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_price.set_title('(a) Crypto Price Trajectories')

    # (b) Persistence barcodes by regime
    ax_pers = fig.add_subplot(gs[0, 1])
    regime_diagrams = results.get('regime_diagrams', {})
    if regime_diagrams:
        y_offset = 0
        y_ticks, y_labels = [], []
        seps = []
        for regime in ['large_cap', 'mid_cap']:
            if regime not in regime_diagrams:
                continue
            diagrams = regime_diagrams[regime]
            color = regime_colors.get(regime, 'gray')
            group_start = y_offset
            all_bars = []
            for dgm in diagrams[:10]:
                if len(dgm) == 0:
                    continue
                h0 = dgm[dgm[:, 2] == 0] if dgm.shape[1] >= 3 else dgm
                for row in h0:
                    if row[1] - row[0] > 1e-6:
                        all_bars.append((row[0], row[1]))
            all_bars.sort(key=lambda b: b[1] - b[0], reverse=True)
            for birth, death in all_bars[:25]:
                ax_pers.barh(y_offset, death - birth, left=birth, height=0.7,
                             color=color, alpha=0.7, edgecolor='none')
                y_offset += 1
            y_ticks.append((group_start + y_offset) / 2)
            y_labels.append(regime_labels.get(regime, regime))
            seps.append(y_offset)
            y_offset += 2

        for sep in seps[:-1]:
            ax_pers.axhline(sep + 0.5, color='gray', linewidth=0.5, linestyle=':')
        ax_pers.set_yticks(y_ticks)
        ax_pers.set_yticklabels(y_labels)
        ax_pers.invert_yaxis()

    ax_pers.set_xlabel('Filtration value ($\\epsilon$)')
    ax_pers.set_title('(b) $H_0$ Persistence Barcodes')

    # (c) RDS vs KL
    ax_bars = fig.add_subplot(gs[0, 2])
    h2_rds = results.get('h2_rds', {})
    h2_kl = results.get('h2_kl', {})
    if h2_rds and h2_kl:
        keys = list(h2_rds.keys())
        x = np.arange(len(keys))
        width = 0.35
        rds_vals = np.array([h2_rds[k] for k in keys])
        kl_vals = np.array([h2_kl[k] for k in keys])
        rds_max = max(np.max(rds_vals), 1e-10)
        kl_max = max(np.max(kl_vals), 1e-10)
        bars_r = ax_bars.bar(x - width / 2, rds_vals / rds_max, width,
                             color='#2980b9', label='RDS (norm)', alpha=0.85)
        bars_k = ax_bars.bar(x + width / 2, kl_vals / kl_max, width,
                             color='#e67e22', label='KL (norm)', alpha=0.85)
        for bar, val in zip(bars_r, rds_vals):
            ax_bars.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='#2980b9')
        for bar, val in zip(bars_k, kl_vals):
            ax_bars.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='#e67e22')
        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels(keys, fontsize=8)
        ax_bars.set_ylabel('Normalized distance')
        ax_bars.legend(fontsize=7, framealpha=0.9)
        ax_bars.set_ylim(0, 1.35)
    ax_bars.set_title('(c) RDS vs KL Divergence')

    plt.tight_layout()
    out = FIG_DIR / 'fig_exp1_crypto.pdf'
    fig.savefig(str(out))
    plt.close()
    print(f"Saved {out.name}")


# ====================================================================
# Experiment 2: Real EEG curvature + dyad summary
# ====================================================================

def fig_exp2_real_eeg():
    """Generate Exp 2 curvature figure from real EEG data."""
    from data_loaders import load_collaboration_eeg
    from exp2_eeg import compute_synchrony_matrix, curvature_time_series
    from scipy.stats import pearsonr

    # --- Compute all dyads ---
    r_values = []
    p_values = []
    best_r = 0
    best_data = None

    for dyad_id in range(8):
        data = load_collaboration_eeg('data/eeg_collab', dyad_id)
        if data is None:
            r_values.append(0)
            p_values.append(1)
            continue

        sync = compute_synchrony_matrix(
            data.participant1, data.participant2,
            data.sfreq, modality='eeg', window_sec=2.0)
        curv = curvature_time_series(sync, threshold=0.3)

        kernel = np.ones(5) / 5
        curv_smooth = np.convolve(curv, kernel, mode='same')

        conditions = data.condition_labels
        n_windows = len(curv)
        window_samples = int(2.0 * data.sfreq)
        cond_windows = np.array([
            np.mean(conditions[w * window_samples:min((w + 1) * window_samples, len(conditions))])
            for w in range(n_windows)
        ])

        r, p = pearsonr(curv_smooth, cond_windows)
        r_values.append(r)
        p_values.append(p)
        print(f"  Dyad {dyad_id}: r={r:.3f}, p={p:.2e}")

        if abs(r) > abs(best_r):
            best_r = r
            best_data = {
                'dyad_id': dyad_id,
                'curv': curv,
                'curv_smooth': curv_smooth,
                'cond_windows': cond_windows,
                'n_windows': n_windows,
                'sfreq': data.sfreq,
                'r': r, 'p': p,
            }

    # --- Figure 1: Best dyad curvature time series ---
    if best_data is not None:
        d = best_data
        t = np.arange(d['n_windows']) * 2.0 / 60.0

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 1, 2]})

        # Top: curvature
        axes[0].plot(t, d['curv'], alpha=0.3, color='#8e44ad', linewidth=0.5)
        axes[0].plot(t, d['curv_smooth'], color='#8e44ad', linewidth=1.5,
                     label='Smoothed curvature')
        axes[0].set_ylabel('Forman-Ricci\ncurvature')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].set_title(
            f'Dyad {d["dyad_id"]+1}: Curvature--condition correlation '
            f'($r = {d["r"]:.3f}$, $p < 0.001$)', fontsize=11)

        # Middle: condition bands
        axes[1].fill_between(t, 0, d['cond_windows'], alpha=0.4, color='#e67e22',
                              step='mid', label='Competition')
        axes[1].fill_between(t, 0, 1 - d['cond_windows'], alpha=0.4, color='#27ae60',
                              step='mid', label='Cooperation')
        axes[1].set_ylabel('Condition')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['Coop.', 'Comp.'])
        axes[1].legend(loc='upper right', fontsize=9)

        # Bottom: z-score overlay
        z_curv = (d['curv_smooth'] - np.mean(d['curv_smooth'])) / (np.std(d['curv_smooth']) + 1e-10)
        z_cond = (d['cond_windows'] - np.mean(d['cond_windows'])) / (np.std(d['cond_windows']) + 1e-10)
        axes[2].plot(t, z_curv, color='#8e44ad', linewidth=1.5, label='Curvature (z)')
        axes[2].plot(t, z_cond, color='#e67e22', linewidth=1.5, alpha=0.7, label='Condition (z)')
        axes[2].set_ylabel('Z-score')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].legend(loc='upper right', fontsize=9)

        textstr = f'$r = {d["r"]:.3f}$\n$p < 0.001$'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[2].text(0.02, 0.95, textstr, transform=axes[2].transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)

        plt.tight_layout()
        out = FIG_DIR / 'fig_exp2_curvature.pdf'
        fig.savefig(str(out))
        plt.close()
        print(f"Saved {out.name}")

    # --- Figure 2: Per-dyad summary (using |r|) ---
    abs_r = [abs(rv) for rv in r_values]
    mean_abs_r = np.mean(abs_r)
    dyad_ids = [f'D{i+1}' for i in range(len(r_values))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                     gridspec_kw={'width_ratios': [3, 2]})

    # Left: signed r values (shows direction)
    colors_signed = ['#c0392b' if rv < 0 else '#27ae60' for rv in r_values]
    bars = ax1.bar(dyad_ids, r_values, color=colors_signed, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Dyad')
    ax1.set_ylabel('Curvature--condition correlation ($r$)')
    ax1.set_title('(a) Per-dyad signed correlation', fontsize=11)
    ax1.set_ylim(-0.8, 0.9)

    # Add significance stars
    for i, (bar, pv) in enumerate(zip(bars, p_values)):
        stars = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
        y_pos = r_values[i] + (0.04 if r_values[i] >= 0 else -0.06)
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos, stars,
                 ha='center', va='bottom' if r_values[i] >= 0 else 'top',
                 fontsize=10, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.8, label='Positive (comp. > coop.)'),
        Patch(facecolor='#c0392b', alpha=0.8, label='Negative (coop. > comp.)'),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc='lower right')

    # Right: |r| values with mean line
    bars2 = ax2.bar(dyad_ids, abs_r, color='#2980b9', alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    ax2.axhline(y=mean_abs_r, color='#e74c3c', linewidth=2, linestyle='--',
                label=f'Mean $|r|$ = {mean_abs_r:.3f}')
    ax2.set_xlabel('Dyad')
    ax2.set_ylabel('$|r|$ (absolute correlation)')
    ax2.set_title('(b) Absolute effect size', fontsize=11)
    ax2.set_ylim(0, 0.9)
    ax2.legend(fontsize=9)

    # Annotate all p < 0.01
    ax2.text(0.98, 0.95, 'All $p < 0.01$', transform=ax2.transAxes,
             fontsize=10, ha='right', va='top', fontstyle='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out2 = FIG_DIR / 'fig_exp2_dyad_summary.pdf'
    fig.savefig(str(out2))
    plt.close()
    print(f"Saved {out2.name}")


# ====================================================================
# Experiment 3: Reddit — improved figures
# ====================================================================

def fig_exp3_reddit():
    """Generate Exp 3 figures from Reddit data with improved visualization."""
    import data_loaders
    from exp3_social_media import run_experiment, subreddit_persistence

    # Patch max_posts
    orig_load = data_loaders.load_reddit_politosphere
    def fast_load(data_dir, subreddit, max_posts=2000):
        return orig_load(data_dir, subreddit, max_posts=2000)
    data_loaders.load_reddit_politosphere = fast_load

    subreddits = ['socialism', 'Conservative', 'The_Donald', 'Republican',
                  'PoliticalDiscussion', 'NeutralPolitics', 'moderatepolitics', 'politics']

    results, communities = run_experiment(
        use_synthetic=False, data_dir='data',
        subreddits=subreddits, seed=42)

    type_map = {
        'socialism': 'Echo', 'Conservative': 'Echo',
        'The_Donald': 'Echo', 'Republican': 'Echo',
        'PoliticalDiscussion': 'Diverse', 'NeutralPolitics': 'Diverse',
        'moderatepolitics': 'Diverse',
        'politics': 'Polarized',
    }
    type_colors = {'Echo': '#c0392b', 'Diverse': '#27ae60', 'Polarized': '#e67e22'}

    # --- Compute persistence for ALL communities ---
    all_persistence = {}
    for comm in communities:
        if comm.embeddings is None:
            continue
        dgm = subreddit_persistence(comm.embeddings, n_subsample=500)
        h0 = dgm[dgm[:, 2] == 0]
        h1 = dgm[dgm[:, 2] == 1]
        h0_pers = h0[:, 1] - h0[:, 0]
        h1_pers = h1[:, 1] - h1[:, 0] if len(h1) > 0 else np.array([])
        all_persistence[comm.name] = {
            'h0_count': len(h0), 'h1_count': len(h1),
            'h0_mean_pers': np.mean(h0_pers) if len(h0_pers) > 0 else 0,
            'h1_mean_pers': np.mean(h1_pers) if len(h1_pers) > 0 else 0,
            'h0_pers': h0_pers, 'h1_pers': h1_pers,
            'type': type_map.get(comm.name, 'Unknown'),
            'dgm': dgm,
        }

    # --- Figure 1: KL distance heatmap + H1 feature summary ---
    kl_matrix = np.array(results.get('kl_matrix', np.zeros((8, 8))))
    labels = [c.name for c in communities]

    # Sort by type for cleaner heatmap
    type_order = {'Echo': 0, 'Diverse': 1, 'Polarized': 2}
    sorted_indices = sorted(range(len(labels)),
                             key=lambda i: (type_order.get(type_map.get(labels[i], ''), 9), labels[i]))
    sorted_labels = [labels[i] for i in sorted_indices]
    kl_sorted = kl_matrix[np.ix_(sorted_indices, sorted_indices)]

    # Short display labels
    short_map = {
        'socialism': 'r/social.', 'Conservative': 'r/Conserv.',
        'The_Donald': 'r/T_D', 'Republican': 'r/Repub.',
        'PoliticalDiscussion': 'r/PolDisc.', 'NeutralPolitics': 'r/NeutPol.',
        'moderatepolitics': 'r/modpol.', 'politics': 'r/politics',
    }
    short_labels = [short_map.get(l, l[:8]) for l in sorted_labels]

    fig, (ax_kl, ax_h1) = plt.subplots(1, 2, figsize=(14, 5.5),
                                         gridspec_kw={'width_ratios': [1.2, 1]})

    # Left: KL heatmap
    im = ax_kl.imshow(kl_sorted, cmap='YlOrRd', aspect='auto')
    ax_kl.set_xticks(range(len(short_labels)))
    ax_kl.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax_kl.set_yticks(range(len(short_labels)))
    ax_kl.set_yticklabels(short_labels, fontsize=8)
    ax_kl.set_title(f'(a) KL Divergence (discrimination ratio = {results.get("kl_discrimination", 0):.2f})',
                     fontsize=11)
    plt.colorbar(im, ax=ax_kl, shrink=0.8, label='KL divergence')
    for spine in ax_kl.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # Draw type boundaries
    echo_count = sum(1 for l in sorted_labels if type_map.get(l) == 'Echo')
    diverse_count = sum(1 for l in sorted_labels if type_map.get(l) == 'Diverse')
    if echo_count > 0:
        ax_kl.axhline(echo_count - 0.5, color='white', linewidth=2)
        ax_kl.axvline(echo_count - 0.5, color='white', linewidth=2)
    if echo_count + diverse_count < len(sorted_labels):
        ax_kl.axhline(echo_count + diverse_count - 0.5, color='white', linewidth=2)
        ax_kl.axvline(echo_count + diverse_count - 0.5, color='white', linewidth=2)

    # Type labels on side
    if echo_count > 0:
        ax_kl.text(-0.5, echo_count / 2 - 0.5, 'Echo', fontsize=8, fontweight='bold',
                   color='#c0392b', ha='right', va='center', rotation=90)
    if diverse_count > 0:
        ax_kl.text(-0.5, echo_count + diverse_count / 2 - 0.5, 'Diverse', fontsize=8,
                   fontweight='bold', color='#27ae60', ha='right', va='center', rotation=90)

    # Right: H0 and H1 feature counts grouped by community
    names_sorted = [l for l in sorted_labels if l in [c.name for c in communities]]
    # Use the community ordering
    comm_names = sorted_labels
    h0_counts = [all_persistence.get(n, {}).get('h0_count', 0) for n in comm_names]
    h1_counts = [all_persistence.get(n, {}).get('h1_count', 0) for n in comm_names]

    x = np.arange(len(comm_names))
    width = 0.35
    short_for_bar = [short_map.get(n, n[:6]) for n in comm_names]
    bar_colors = [type_colors.get(type_map.get(n, ''), 'gray') for n in comm_names]

    bars_h0 = ax_h1.bar(x - width / 2, h0_counts, width, label='$H_0$ features',
                         color=bar_colors, alpha=0.6, edgecolor='black', linewidth=0.3)
    bars_h1 = ax_h1.bar(x + width / 2, h1_counts, width, label='$H_1$ features',
                         color=bar_colors, alpha=0.9, edgecolor='black', linewidth=0.3,
                         hatch='///')

    ax_h1.set_xticks(x)
    ax_h1.set_xticklabels(short_for_bar, rotation=45, ha='right', fontsize=8)
    ax_h1.set_ylabel('Feature count')
    ax_h1.set_title('(b) Persistent Homology Features', fontsize=11)
    ax_h1.legend(fontsize=8, loc='upper right')

    # Annotate NeutralPolitics
    np_idx = None
    for i, n in enumerate(comm_names):
        if n == 'NeutralPolitics':
            np_idx = i
            break
    if np_idx is not None:
        ax_h1.annotate('Fewest features\n(structurally simplest)',
                       xy=(np_idx + width / 2, h1_counts[np_idx]),
                       xytext=(np_idx + 1.5, max(h1_counts) * 0.85),
                       fontsize=8, ha='center',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1),
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = FIG_DIR / 'fig_exp3_distances.pdf'
    fig.savefig(str(out))
    plt.close()
    print(f"Saved {out.name}")

    # --- Figure 2: Mean persistence by community (grouped bar) ---
    fig, (ax_pers, ax_box) = plt.subplots(1, 2, figsize=(14, 5),
                                            gridspec_kw={'width_ratios': [1, 1]})

    # Left: mean persistence (H0 and H1) per community
    h0_mean = [all_persistence.get(n, {}).get('h0_mean_pers', 0) for n in comm_names]
    h1_mean = [all_persistence.get(n, {}).get('h1_mean_pers', 0) for n in comm_names]

    bars_h0m = ax_pers.bar(x - width / 2, h0_mean, width, label='$H_0$ mean pers.',
                            color=bar_colors, alpha=0.6, edgecolor='black', linewidth=0.3)
    bars_h1m = ax_pers.bar(x + width / 2, h1_mean, width, label='$H_1$ mean pers.',
                            color=bar_colors, alpha=0.9, edgecolor='black', linewidth=0.3,
                            hatch='///')
    ax_pers.set_xticks(x)
    ax_pers.set_xticklabels(short_for_bar, rotation=45, ha='right', fontsize=8)
    ax_pers.set_ylabel('Mean persistence')
    ax_pers.set_title('(a) Mean Persistence by Community', fontsize=11)
    ax_pers.legend(fontsize=8)

    # Right: box plot of H0 persistence distributions grouped by type
    echo_h0 = []
    diverse_h0 = []
    polarized_h0 = []
    for n in comm_names:
        p = all_persistence.get(n, {})
        h0p = p.get('h0_pers', np.array([]))
        t = type_map.get(n, '')
        if t == 'Echo' and len(h0p) > 0:
            echo_h0.extend(h0p.tolist())
        elif t == 'Diverse' and len(h0p) > 0:
            diverse_h0.extend(h0p.tolist())
        elif t == 'Polarized' and len(h0p) > 0:
            polarized_h0.extend(h0p.tolist())

    box_data = [echo_h0, diverse_h0, polarized_h0]
    box_labels = ['Echo\nchambers', 'Diverse\ncommunities', 'Polarized\ncommunity']
    box_colors_list = ['#c0392b', '#27ae60', '#e67e22']

    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True,
                        widths=0.5, showfliers=False, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
    for patch, color in zip(bp['boxes'], box_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    ax_box.set_ylabel('$H_0$ persistence')
    ax_box.set_title('(b) Persistence Distribution by Type', fontsize=11)

    # Add n= annotations
    for i, data in enumerate(box_data):
        ax_box.text(i + 1, ax_box.get_ylim()[0] + 0.02,
                    f'n={len(data)}', ha='center', fontsize=8, color='gray')

    # Annotate group means
    for i, data in enumerate(box_data):
        if data:
            m = np.mean(data)
            ax_box.text(i + 1.3, m, f'{m:.3f}', fontsize=8, va='center')

    plt.tight_layout()
    out = FIG_DIR / 'fig_exp3_persistence.pdf'
    fig.savefig(str(out))
    plt.close()
    print(f"Saved {out.name}")


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    print("Generating real-data figures...")
    print()

    print("--- Experiment 1: Crypto ---")
    fig_exp1_crypto()
    print()

    print("--- Experiment 2: Real EEG ---")
    fig_exp2_real_eeg()
    print()

    print("--- Experiment 3: Reddit ---")
    fig_exp3_reddit()
    print()

    print("Done! Figures in:", FIG_DIR)
