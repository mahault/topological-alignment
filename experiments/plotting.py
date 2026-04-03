"""
Plotting utilities for Belief Geodesics Framework paper.
=========================================================

Generates publication-quality figures for all three experiments.

Figures produced:
  1. fig_framework_overview.pdf  — Conceptual belief landscape with basins, geodesic, ridge
  2. fig_exp1_combined.pdf       — 4-panel: trajectories, persistence barcodes, RDS/KL bars, empowerment
  3. fig_exp2_curvature.pdf      — 3-panel: curvature, coupling, overlay (oversampled, shaded epochs)
  4. fig_exp3_distances.pdf      — Sorted distance heatmaps (RDS, KL, embedding) with diverging cmap
  5. fig_exp3_persistence.pdf    — Persistence barcodes for echo/diverse/polarized communities
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.interpolate import make_interp_spline

# ====================================================================
# Publication style
# ====================================================================

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

COLORS = {
    'R': '#c0392b',    # deep red -- rigid
    'F': '#27ae60',    # green -- flexible
    'M': '#f39c12',    # amber/gold -- mixed
    'echo': '#c0392b',
    'diverse': '#27ae60',
    'polarized': '#e67e22',
    'curvature': '#8e44ad',
    'coupling': '#2c3e50',
    'rds': '#2980b9',
    'kl': '#e67e22',
    'emb': '#27ae60',
}

GROUP_LABELS = {'R': 'Rigid', 'F': 'Flexible', 'M': 'Mixed'}

FIG_DIR = Path(__file__).parent.parent / 'figures'


def ensure_fig_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _covariance_ellipse(points, n_std=2.0):
    """
    Compute the n_std-sigma covariance ellipse for a set of 2D points.

    Returns (center_x, center_y, width, height, angle_degrees).
    Pure numpy -- no sklearn.
    """
    if len(points) < 3:
        return None
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)
    if cov.ndim < 2:
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_std * np.sqrt(max(eigvals[0], 0))
    height = 2 * n_std * np.sqrt(max(eigvals[1], 0))
    return mean[0], mean[1], width, height, angle


def _smooth_oversample(x, y, factor=2):
    """
    Oversample a time series by the given factor using cubic B-spline
    interpolation.  Returns (x_new, y_new).
    """
    if len(x) < 4:
        return x, y
    n_new = len(x) * factor
    x_new = np.linspace(x[0], x[-1], n_new)
    try:
        spl = make_interp_spline(x, y, k=3)
        y_new = spl(x_new)
    except Exception:
        y_new = np.interp(x_new, x, y)
    return x_new, y_new


# ====================================================================
# Figure 1: Framework Overview -- Belief Landscape Schematic
# ====================================================================

def fig_framework_overview(filename: str = 'fig_framework_overview.pdf'):
    """
    Conceptual figure: 2D belief landscape with two attractor basins,
    a surprise geodesic connecting them, and a ridge between them.
    """
    ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Build a 2D potential landscape with two basins
    # Basin A centered at (-2.5, 0), Basin B centered at (2.5, 0.5)
    xg = np.linspace(-6, 6, 400)
    yg = np.linspace(-4, 4, 300)
    X, Y = np.meshgrid(xg, yg)

    # Two Gaussian wells + a ridge
    basin_a = (-2.5, 0.0)
    basin_b = (2.5, 0.5)
    sigma_a = 1.6
    sigma_b = 1.8

    Za = np.exp(-((X - basin_a[0])**2 + (Y - basin_a[1])**2) / (2 * sigma_a**2))
    Zb = np.exp(-((X - basin_b[0])**2 + (Y - basin_b[1])**2) / (2 * sigma_b**2))

    # Free energy landscape (inverted wells = low free energy at basins)
    F = -2.0 * Za - 1.8 * Zb + 0.3 * np.exp(-((X)**2) / 1.5) * np.exp(-(Y**2) / 6.0)
    # Normalize for display
    F = (F - F.min()) / (F.max() - F.min())

    # Contour plot
    levels = np.linspace(0, 1, 25)
    cf = ax.contourf(X, Y, F, levels=levels, cmap='cividis_r', alpha=0.85)
    ax.contour(X, Y, F, levels=levels[::3], colors='white', linewidths=0.3, alpha=0.4)

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Free Energy $F(\\phi)$', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Mark basin centers
    ax.plot(*basin_a, 'o', color='white', markersize=10, markeredgecolor='black',
            markeredgewidth=1.2, zorder=10)
    ax.plot(*basin_b, 's', color='white', markersize=10, markeredgecolor='black',
            markeredgewidth=1.2, zorder=10)

    # Labels for basins
    ax.annotate('Basin A\n(Agent 1)', xy=basin_a, xytext=(-4.5, -2.5),
                fontsize=10, fontweight='bold', color='white',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#2c3e50', alpha=0.8))

    ax.annotate('Basin B\n(Agent 2)', xy=basin_b, xytext=(4.5, -2.2),
                fontsize=10, fontweight='bold', color='white',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#2c3e50', alpha=0.8))

    # Geodesic path between basins (curved to go through ridge saddle)
    t_geo = np.linspace(0, 1, 200)
    # Parametric path that bends upward to cross the ridge at its lowest point
    geo_x = basin_a[0] + (basin_b[0] - basin_a[0]) * t_geo
    # Bend upward through the saddle point
    geo_y = basin_a[1] + (basin_b[1] - basin_a[1]) * t_geo + 1.2 * np.sin(np.pi * t_geo)

    # Color the geodesic by local free energy
    points = np.array([geo_x, geo_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Evaluate free energy along path
    geo_Za = np.exp(-((geo_x - basin_a[0])**2 + (geo_y - basin_a[1])**2) / (2 * sigma_a**2))
    geo_Zb = np.exp(-((geo_x - basin_b[0])**2 + (geo_y - basin_b[1])**2) / (2 * sigma_b**2))
    geo_F = -2.0 * geo_Za - 1.8 * geo_Zb + 0.3 * np.exp(-(geo_x**2) / 1.5) * np.exp(-(geo_y**2) / 6.0)
    geo_F_norm = (geo_F - geo_F.min()) / (geo_F.max() - geo_F.min() + 1e-10)

    lc = LineCollection(segments, cmap='hot_r', linewidth=3, zorder=8)
    lc.set_array(geo_F_norm[:-1])
    ax.add_collection(lc)

    # Arrow at midpoint of geodesic
    mid = len(t_geo) // 2
    dx = geo_x[mid + 5] - geo_x[mid - 5]
    dy = geo_y[mid + 5] - geo_y[mid - 5]
    ax.annotate('', xy=(geo_x[mid] + dx * 0.3, geo_y[mid] + dy * 0.3),
                xytext=(geo_x[mid] - dx * 0.1, geo_y[mid] - dy * 0.1),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5),
                zorder=9)

    # Label the geodesic
    ax.text(0.0, 2.2, 'Surprise Geodesic', fontsize=10, fontweight='bold',
            color='#e74c3c', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#e74c3c'))

    # Label the ridge
    ax.text(0.0, -1.3, 'Ridge', fontsize=9, fontstyle='italic',
            color='white', ha='center', va='center', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#7f8c8d', alpha=0.7))

    # Axis labels
    ax.set_xlabel(r'Belief coordinate $\phi_1$', fontsize=11)
    ax.set_ylabel(r'Belief coordinate $\phi_2$', fontsize=11)
    ax.set_title('Belief Landscape: Attractor Basins and Surprise Geodesic',
                 fontsize=12, fontweight='bold', pad=12)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    # Re-enable spines for this figure (contour looks better boxed)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


# ====================================================================
# Figure 2: Experiment 1 Combined (4-panel)
# ====================================================================

def fig_exp1_combined(trajectories: np.ndarray, group_indices: dict,
                      rds_dists: dict, kl_dists: dict,
                      empowerment: dict, group_diagrams: dict = None,
                      filename: str = 'fig_exp1_combined.pdf'):
    """
    Combined 4-panel figure for Experiment 1.

    (a) Belief trajectories with 2-sigma covariance ellipses
    (b) Persistence barcodes by group
    (c) RDS vs KL grouped bar chart
    (d) Empowerment by group
    """
    ensure_fig_dir()

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ------------------------------------------------------------------
    # Panel (a): Trajectories with attractor ellipses
    # ------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[0, 0])

    # Plot each group with distinct styling
    style_map = {
        'R': dict(alpha_line=0.08, alpha_fill=0.25, lw=0.4, linestyle='-'),
        'F': dict(alpha_line=0.04, alpha_fill=0.12, lw=0.3, linestyle='-'),
        'M': dict(alpha_line=0.06, alpha_fill=0.18, lw=0.35, linestyle='-'),
    }

    for name, idx in group_indices.items():
        color = COLORS.get(name, 'gray')
        style = style_map.get(name, style_map['M'])
        label = GROUP_LABELS.get(name, name)

        # Plot individual trajectories (subsample for clarity)
        n_show = min(40, len(idx))
        for i in idx[:n_show]:
            traj = trajectories[:, i, :2]
            ax_traj.plot(traj[:, 0], traj[:, 1],
                         alpha=style['alpha_line'], linewidth=style['lw'],
                         color=color, linestyle=style['linestyle'])

        # Centroid trajectory
        centroid = np.mean(trajectories[:, idx, :2], axis=1)
        ax_traj.plot(centroid[:, 0], centroid[:, 1], color=color,
                     linewidth=2.0, label=label, zorder=5)

        # 2-sigma covariance ellipse on final beliefs
        final_beliefs = trajectories[-1, idx, :2]
        ell_params = _covariance_ellipse(final_beliefs, n_std=2.0)
        if ell_params is not None:
            cx, cy, w, h, angle = ell_params
            ellipse = Ellipse((cx, cy), w, h, angle=angle,
                              facecolor=color, edgecolor=color,
                              alpha=style['alpha_fill'], linewidth=1.5,
                              linestyle='--', zorder=4)
            ax_traj.add_patch(ellipse)

    ax_traj.set_xlabel(r'$\phi_1$')
    ax_traj.set_ylabel(r'$\phi_2$')
    ax_traj.set_title('(a) Belief Trajectories + 2$\\sigma$ Attractors', fontsize=11)
    ax_traj.legend(loc='upper right', framealpha=0.9)

    # ------------------------------------------------------------------
    # Panel (b): Persistence barcodes
    # ------------------------------------------------------------------
    ax_pers = fig.add_subplot(gs[0, 1])

    if group_diagrams is not None:
        y_offset = 0
        y_ticks = []
        y_labels = []
        group_separators = []

        for name in ['R', 'F', 'M']:
            if name not in group_diagrams:
                continue
            diagrams = group_diagrams[name]
            color = COLORS.get(name, 'gray')
            label = GROUP_LABELS.get(name, name)
            group_start = y_offset

            # Collect all persistence intervals from first few diagrams
            all_bars = []
            for dgm in diagrams[:10]:
                if len(dgm) == 0:
                    continue
                h0 = dgm[dgm[:, 2] == 0] if dgm.shape[1] >= 3 else dgm
                for row in h0:
                    if row[1] - row[0] > 1e-6:
                        all_bars.append((row[0], row[1]))

            # Sort by persistence (death - birth) descending, show top 25
            all_bars.sort(key=lambda b: b[1] - b[0], reverse=True)
            bars_to_show = all_bars[:25]

            for birth, death in bars_to_show:
                ax_pers.barh(y_offset, death - birth, left=birth, height=0.7,
                             color=color, alpha=0.7, edgecolor='none')
                y_offset += 1

            y_ticks.append((group_start + y_offset) / 2)
            y_labels.append(label)
            group_separators.append(y_offset)
            y_offset += 2  # gap between groups

        # Horizontal separators
        for sep in group_separators[:-1]:
            ax_pers.axhline(sep + 0.5, color='gray', linewidth=0.5, linestyle=':')

        ax_pers.set_yticks(y_ticks)
        ax_pers.set_yticklabels(y_labels)
        ax_pers.set_xlabel('Filtration value ($\\epsilon$)')
        ax_pers.set_title('(b) $H_0$ Persistence Barcodes', fontsize=11)
        ax_pers.invert_yaxis()
    else:
        ax_pers.text(0.5, 0.5, 'No persistence data', ha='center', va='center',
                     transform=ax_pers.transAxes, fontsize=11, color='gray')
        ax_pers.set_title('(b) $H_0$ Persistence Barcodes', fontsize=11)

    # ------------------------------------------------------------------
    # Panel (c): RDS vs KL grouped bar chart
    # ------------------------------------------------------------------
    ax_bars = fig.add_subplot(gs[1, 0])

    keys = list(rds_dists.keys())
    x = np.arange(len(keys))
    width = 0.35

    rds_vals = np.array([rds_dists[k] for k in keys])
    kl_vals = np.array([kl_dists[k] for k in keys])

    # Normalize both to [0, 1] for comparable display
    rds_max = np.max(rds_vals) if np.max(rds_vals) > 0 else 1.0
    kl_max = np.max(kl_vals) if np.max(kl_vals) > 0 else 1.0

    bars_rds = ax_bars.bar(x - width / 2, rds_vals / rds_max, width,
                            color=COLORS['rds'], label='RDS distance (norm)',
                            edgecolor='white', linewidth=0.5, alpha=0.85)
    bars_kl = ax_bars.bar(x + width / 2, kl_vals / kl_max, width,
                           color=COLORS['kl'], label='KL divergence (norm)',
                           edgecolor='white', linewidth=0.5, alpha=0.85)

    # Value labels
    for bar, val in zip(bars_rds, rds_vals):
        ax_bars.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7, color=COLORS['rds'])
    for bar, val in zip(bars_kl, kl_vals):
        ax_bars.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7, color=COLORS['kl'])

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(keys, fontsize=9)
    ax_bars.set_ylabel('Normalized distance')
    ax_bars.set_title('(c) RDS Distance vs KL Divergence', fontsize=11)
    ax_bars.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_bars.set_ylim(0, 1.35)

    # ------------------------------------------------------------------
    # Panel (d): Empowerment by group
    # ------------------------------------------------------------------
    ax_emp = fig.add_subplot(gs[1, 1])

    group_names = list(empowerment.keys())
    emp_vals = [empowerment[k] for k in group_names]
    emp_colors = [COLORS.get(k, 'gray') for k in group_names]
    emp_labels = [GROUP_LABELS.get(k, k) for k in group_names]

    bars_emp = ax_emp.bar(emp_labels, emp_vals, color=emp_colors, width=0.5,
                           edgecolor='white', linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars_emp, emp_vals):
        ax_emp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax_emp.set_ylabel('Empowerment')
    ax_emp.set_title('(d) Geometric Empowerment by Group', fontsize=11)
    # Add annotation
    ax_emp.annotate('Higher = more reachable\nbelief states',
                    xy=(0.98, 0.95), xycoords='axes fraction',
                    fontsize=7, color='gray', ha='right', va='top',
                    fontstyle='italic')

    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


# ====================================================================
# Figure 3: Experiment 2 -- EEG Curvature (improved)
# ====================================================================

def fig_curvature_coupling(curv_ts: np.ndarray, coupling_ts: np.ndarray,
                            sfreq_windows: float = 0.5,
                            filename: str = 'fig_exp2_curvature.pdf'):
    """
    3-panel figure: curvature, coupling, and overlay with vertical
    shading bands for high-coupling vs low-coupling epochs.
    Uses 2x oversampled interpolation for visual smoothness.
    """
    ensure_fig_dir()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

    n_pts = min(len(curv_ts), len(coupling_ts))
    curv_ts = curv_ts[:n_pts]
    coupling_ts = coupling_ts[:n_pts]
    t_raw = np.arange(n_pts) / sfreq_windows

    # 2x oversampled interpolation for visual smoothness
    t_os, curv_os = _smooth_oversample(t_raw, curv_ts, factor=2)
    t_os2, coup_os = _smooth_oversample(t_raw, coupling_ts, factor=2)

    # Determine high-coupling vs low-coupling epochs from raw coupling
    coupling_median = np.median(coupling_ts)
    high_coupling = coupling_ts > coupling_median

    # Add vertical shading bands to all three panels
    def _shade_epochs(ax, t, high_mask, alpha=0.08):
        """Add vertical shading for high-coupling (green) and low-coupling (red) bands."""
        in_high = False
        start = 0
        for i in range(len(high_mask)):
            if high_mask[i] and not in_high:
                start = t[i]
                in_high = True
            elif not high_mask[i] and in_high:
                ax.axvspan(start, t[i], color='#27ae60', alpha=alpha, zorder=0)
                in_high = False
        if in_high:
            ax.axvspan(start, t[-1], color='#27ae60', alpha=alpha, zorder=0)

        in_low = False
        for i in range(len(high_mask)):
            if not high_mask[i] and not in_low:
                start = t[i]
                in_low = True
            elif high_mask[i] and in_low:
                ax.axvspan(start, t[i], color='#c0392b', alpha=alpha * 0.6, zorder=0)
                in_low = False
        if in_low:
            ax.axvspan(start, t[-1], color='#c0392b', alpha=alpha * 0.6, zorder=0)

    for a in [ax1, ax2, ax3]:
        _shade_epochs(a, t_raw, high_coupling, alpha=0.10)

    # ---- Panel 1: Curvature ----
    ax1.plot(t_os, curv_os, color=COLORS['curvature'], linewidth=0.6, alpha=0.5)
    # Moving average (smooth trend)
    kernel_size = min(7, n_pts // 3)
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        curv_smooth = np.convolve(curv_ts, kernel, mode='same')
        t_sm, curv_sm_os = _smooth_oversample(t_raw, curv_smooth, factor=2)
        ax1.plot(t_sm, curv_sm_os, color=COLORS['curvature'], linewidth=2.5, zorder=5)
    ax1.set_ylabel('Forman-Ricci\nCurvature', fontsize=10)
    ax1.set_title('Inter-Brain Curvature Dynamics', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # ---- Panel 2: Coupling ----
    ax2.plot(t_os2, coup_os, color=COLORS['coupling'], linewidth=1.8, zorder=5)
    ax2.axhline(coupling_median, color='gray', linewidth=0.8, linestyle='--', alpha=0.5,
                label=f'Median = {coupling_median:.3f}')
    ax2.set_ylabel('Coupling\nStrength', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # ---- Panel 3: Overlay (z-scored) ----
    curv_z = (curv_ts - np.mean(curv_ts)) / (np.std(curv_ts) + 1e-8)
    coup_z = (coupling_ts - np.mean(coupling_ts)) / (np.std(coupling_ts) + 1e-8)
    t_cz, curv_z_os = _smooth_oversample(t_raw, curv_z, factor=2)
    t_cpz, coup_z_os = _smooth_oversample(t_raw, coup_z, factor=2)

    ax3.plot(t_cz, curv_z_os, color=COLORS['curvature'], linewidth=1.2, alpha=0.7,
             label='Curvature (z)')
    ax3.plot(t_cpz, coup_z_os, color=COLORS['coupling'], linewidth=1.8,
             label='Coupling (z)')
    ax3.axhline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)
    ax3.set_ylabel('z-score', fontsize=10)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Legend for shading (add to panel 1)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.2, label='High coupling'),
        Patch(facecolor='#c0392b', alpha=0.12, label='Low coupling'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


# ====================================================================
# Figure 4: Experiment 3 -- Distance Matrices (improved)
# ====================================================================

def fig_distance_matrices(rds_matrix: np.ndarray, kl_matrix: np.ndarray,
                           emb_matrix: np.ndarray, names: list,
                           filename: str = 'fig_exp3_distances.pdf'):
    """
    Distance heatmaps with communities sorted by type, short labels,
    and a diverging colormap.
    """
    ensure_fig_dir()

    # Classify communities by type
    type_order = {'echo': 0, 'diverse': 1, 'polarized': 2}
    type_prefix = {'echo_chamber': 'Echo', 'echo': 'Echo',
                   'diverse': 'Div', 'polarized': 'Pol'}

    def _get_type(name):
        name_lower = name.lower()
        if 'echo' in name_lower:
            return 'echo'
        elif 'diverse' in name_lower:
            return 'diverse'
        elif 'polarized' in name_lower or 'polar' in name_lower:
            return 'polarized'
        return 'other'

    # Sort indices so same-type communities are adjacent
    indices_with_type = []
    type_counters = {'echo': 0, 'diverse': 0, 'polarized': 0, 'other': 0}
    for i, name in enumerate(names):
        t = _get_type(name)
        type_counters[t] += 1
        indices_with_type.append((i, t, type_counters[t]))

    sorted_info = sorted(indices_with_type, key=lambda x: (type_order.get(x[1], 99), x[2]))
    sort_idx = [x[0] for x in sorted_info]

    # Short labels
    short_names = []
    for _, t, c in sorted_info:
        prefix = type_prefix.get(t, t[:3].capitalize())
        short_names.append(f'{prefix} {c}')

    # Reorder matrices
    rds_sorted = rds_matrix[np.ix_(sort_idx, sort_idx)]
    kl_sorted = kl_matrix[np.ix_(sort_idx, sort_idx)]
    emb_sorted = emb_matrix[np.ix_(sort_idx, sort_idx)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    matrices = [rds_sorted, kl_sorted, emb_sorted]
    titles = ['RDS Distance', 'KL Divergence', 'Embedding Distance']

    for ax, mat, title in zip(axes, matrices, titles):
        # Normalize for display
        mat_max = np.max(mat) if np.max(mat) > 0 else 1.0
        mat_norm = mat / mat_max

        im = ax.imshow(mat_norm, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto',
                        interpolation='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=55, ha='right', fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)

        # Draw type-group boundaries
        cumulative = 0
        for t_name in ['echo', 'diverse', 'polarized']:
            count = sum(1 for _, t, _ in sorted_info if t == t_name)
            if count > 0:
                pos = cumulative + count - 0.5
                if cumulative + count < len(short_names):
                    ax.axhline(pos, color='black', linewidth=1.0, alpha=0.5)
                    ax.axvline(pos, color='black', linewidth=1.0, alpha=0.5)
                cumulative += count

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Re-enable all spines for heatmap
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)
    print(f"Saved {filename}")


# ====================================================================
# Figure 5: Experiment 3 -- Persistence Barcodes by Community Type
# ====================================================================

def fig_exp3_persistence(communities: list, tda_results: dict,
                          filename: str = 'fig_exp3_persistence.pdf'):
    """
    Side-by-side persistence barcodes for one representative community
    of each type: echo chamber, diverse, polarized.

    Highlights how echo chambers have low mean persistence (tight basin)
    vs diverse communities (high persistence, spread attractors).
    """
    ensure_fig_dir()
    from exp1_synthetic import compute_persistence

    # Pick one representative of each type
    type_reps = {'Echo Chamber': None, 'Diverse': None, 'Polarized': None}
    type_keys = {'Echo Chamber': 'echo', 'Diverse': 'diverse', 'Polarized': 'polarized'}

    for comm in communities:
        name_lower = comm.name.lower()
        if 'echo' in name_lower and type_reps['Echo Chamber'] is None:
            type_reps['Echo Chamber'] = comm
        elif 'diverse' in name_lower and type_reps['Diverse'] is None:
            type_reps['Diverse'] = comm
        elif 'polarized' in name_lower and type_reps['Polarized'] is None:
            type_reps['Polarized'] = comm

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    type_colors = {
        'Echo Chamber': COLORS['echo'],
        'Diverse': COLORS['diverse'],
        'Polarized': COLORS['polarized'],
    }

    all_mean_pers = {}

    for ax, (type_name, comm) in zip(axes, type_reps.items()):
        color = type_colors[type_name]
        if comm is None or comm.embeddings is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(type_name)
            continue

        # Compute persistence for this community
        dgm = compute_persistence(comm.embeddings, max_dim=1,
                                   max_edge=15.0, n_subsample=500)

        if len(dgm) == 0:
            ax.text(0.5, 0.5, 'No features', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(type_name)
            continue

        # Filter to H0 features
        h0 = dgm[dgm[:, 2] == 0] if dgm.shape[1] >= 3 else dgm
        if len(h0) == 0:
            ax.text(0.5, 0.5, 'No $H_0$ features', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(type_name)
            continue

        # Sort by persistence descending
        persistence = h0[:, 1] - h0[:, 0]
        order = np.argsort(persistence)[::-1]
        h0_sorted = h0[order]
        pers_sorted = persistence[order]

        # Show top 40 features
        n_show = min(40, len(h0_sorted))
        h0_show = h0_sorted[:n_show]
        pers_show = pers_sorted[:n_show]

        mean_pers = np.mean(pers_show)
        all_mean_pers[type_name] = mean_pers

        # Draw barcodes
        for j in range(n_show):
            birth, death = h0_show[j, 0], h0_show[j, 1]
            alpha_val = 0.4 + 0.6 * (pers_show[j] / (pers_show[0] + 1e-10))
            ax.barh(j, death - birth, left=birth, height=0.8,
                    color=color, alpha=min(1.0, alpha_val), edgecolor='none')

        # Mean persistence line
        ax.axvline(mean_pers, color='black', linewidth=1.2, linestyle='--', alpha=0.7,
                   label=f'Mean pers = {mean_pers:.2f}')

        ax.set_title(f'{type_name}', fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('Filtration value ($\\epsilon$)')
        if ax == axes[0]:
            ax.set_ylabel('Feature rank')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax.invert_yaxis()

    # Add summary annotation
    if len(all_mean_pers) == 3:
        summary = ' | '.join([f'{k}: {v:.2f}' for k, v in all_mean_pers.items()])
        fig.text(0.5, -0.02, f'Mean persistence -- {summary}',
                 ha='center', fontsize=9, fontstyle='italic', color='gray')

    plt.tight_layout()
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

    # Figure 1: Framework overview (no data needed)
    fig_framework_overview()

    # Experiment 1
    print("\n--- Experiment 1 ---")
    from exp1_synthetic import run_experiment as run_exp1
    results1, traj1, groups1 = run_exp1(seed=42, n_steps=10000)

    # Compute persistence diagrams for panel (b)
    from exp1_synthetic import takens_embedding, compute_persistence
    group_diagrams = {}
    for name, idx in groups1.items():
        diagrams = []
        for i in idx[:20]:
            agent_traj = traj1[:, i, :]
            try:
                embedded = takens_embedding(agent_traj, tau=10, d_e=10)
                dgm = compute_persistence(embedded, max_dim=1, n_subsample=300)
                diagrams.append(dgm)
            except Exception:
                pass
        group_diagrams[name] = diagrams

    fig_exp1_combined(
        trajectories=traj1,
        group_indices=groups1,
        rds_dists=results1['h2_rds'],
        kl_dists=results1['h2_kl'],
        empowerment=results1['h5_empowerment'],
        group_diagrams=group_diagrams,
    )

    # Experiment 2
    print("\n--- Experiment 2 ---")
    from exp2_eeg import run_experiment_synthetic as run_exp2
    results2, curv_ts, coupling_ds = run_exp2(seed=42)

    fig_curvature_coupling(curv_ts, coupling_ds)

    # Experiment 3
    print("\n--- Experiment 3 ---")
    from exp3_social_media import run_experiment as run_exp3
    results3, communities = run_exp3(use_synthetic=True, seed=42)

    names = [c.name for c in communities]
    fig_distance_matrices(results3['rds_matrix'], results3['kl_matrix'],
                           results3['emb_matrix'], names)

    fig_exp3_persistence(communities, results3.get('tda_results', {}))

    print("\n" + "=" * 60)
    print("All figures generated.")
    print(f"Output directory: {FIG_DIR}")
    print("Figures:")
    print("  1. fig_framework_overview.pdf")
    print("  2. fig_exp1_combined.pdf")
    print("  3. fig_exp2_curvature.pdf")
    print("  4. fig_exp3_distances.pdf")
    print("  5. fig_exp3_persistence.pdf")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_figures()
