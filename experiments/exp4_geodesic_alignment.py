"""
Experiment 4: Fisher Geodesics + Gromov-Wasserstein Belief-Manifold Alignment
==============================================================================
Tests the geodesic/GW machinery (geodesics.py) on the synthetic belief
network from Experiment 1. Addresses open problems 1 (structural distance
between belief-generating systems) and 3 (least-effort belief transition)
of the extended abstract.

Hypotheses:
  H-G1: GW distance separates rigid/flexible/mixed belief manifolds
        WITHOUT any shared coordinate system (each agent's posterior
        cloud is expressed in a private, randomly-rotated frame).
        A distributional baseline (symmetrized Gaussian KL) computed in
        the same private frames should fail.
  H-G2: Fisher-Rao geodesics between confident beliefs pass through
        increased uncertainty (the geometric form of the anxiety-gated
        inertia->elasticity transition), and the cost of the same belief
        change is far higher under a rigid (low-sigma) metric than a
        flexible one. Empirical Fisher matches the closed form.
  H-G3: Sampling along independently-computed geodesics (endpoint
        correspondence only) + per-step GW localizes WHERE two agents'
        belief dynamics diverge along a belief transition.
  H-G4: Sliding-window GW tracks whether two groups are moving toward
        or away from each other. When the Rigid group is made flexible
        mid-run (obs noise, learning rate and precision set to
        Flexible-like values), GW(R, F) falls; in a no-perturbation
        control it stays flat. Note: a precision-only drop (the exp1 H4
        perturbation) does NOT move GW — it changes drift, not the
        structure of the belief cloud — which is itself evidence that
        GW measures structure rather than update gain.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from exp1_synthetic import (GroupConfig, create_group, build_erdos_renyi,
                            SyntheticBeliefNetwork)
from geodesics import (gaussian_fisher_metric, gaussian_score, empirical_fisher,
                       compute_geodesic, path_energy, path_length,
                       entropic_gromov_wasserstein, gw_misalignment_profile)


# ====================================================================
# Helpers
# ====================================================================

def random_isometry(d: int, rng: np.random.Generator, shift_scale: float = 5.0):
    """A private coordinate frame: random rotation + translation."""
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    t = rng.normal(0, shift_scale, size=d)
    return Q, t


def apply_isometry(points: np.ndarray, Q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ Q.T + t[None, :]


def symmetrized_gaussian_kl(x: np.ndarray, y: np.ndarray) -> float:
    """Symmetrized KL between Gaussian fits of two clouds (coordinate-dependent)."""
    def kl(a, b):
        mu1, cov1 = a.mean(0), np.cov(a.T) + 1e-6 * np.eye(a.shape[1])
        mu2, cov2 = b.mean(0), np.cov(b.T) + 1e-6 * np.eye(b.shape[1])
        d = len(mu1)
        inv2 = np.linalg.inv(cov2)
        diff = mu2 - mu1
        return 0.5 * (np.trace(inv2 @ cov1) + diff @ inv2 @ diff - d
                      + np.log(np.linalg.det(cov2) / np.linalg.det(cov1)))
    return max(kl(x, y) + kl(y, x), 0.0)


def separation_stats(dist_matrix: np.ndarray, labels: np.ndarray,
                     rng: np.random.Generator, n_perm: int = 2000):
    """
    Within- vs between-group mean distances and a label-permutation
    p-value for the statistic (mean_between - mean_within).
    """
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    dists = dist_matrix[iu]
    same = (labels[iu[0]] == labels[iu[1]])

    def stat(same_mask):
        return dists[~same_mask].mean() - dists[same_mask].mean()

    observed = stat(same)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        same_p = (perm[iu[0]] == perm[iu[1]])
        if stat(same_p) >= observed:
            count += 1
    p_val = (count + 1) / (n_perm + 1)
    return {
        'within_mean': float(dists[same].mean()),
        'between_mean': float(dists[~same].mean()),
        'ratio': float(dists[~same].mean() / max(dists[same].mean(), 1e-12)),
        'statistic': float(observed),
        'p_value': float(p_val),
    }


def simulate_network(seed: int = 42, n_steps: int = 6000, d: int = 5,
                     flexibilize: bool = False):
    """
    Exp-1 setup: R/F/M groups. With `flexibilize=True`, the Rigid
    group's parameters are rewritten to Flexible-like values at the
    midpoint (obs_noise_std 0.2 -> 1.5, learning_rate 0.01 -> 0.05,
    precision ~U(8,12) -> ~U(1,3), social_coupling ~U(0.01,0.05) ->
    ~U(0.1,0.3)) — a structural change the GW distance should detect
    as R converging toward F. All four parameters matter: a partial
    change (e.g. without coupling) makes R overshoot F into higher
    diffuseness, which GW also detects — as an INCREASE.
    """
    rng = np.random.default_rng(seed)
    env_signal = rng.normal(0, 1, size=d)

    configs = {
        'R': GroupConfig('Rigid', 170, (8, 12), 0.01, (0.01, 0.05), 0.2, 0.05, 0.1),
        'F': GroupConfig('Flexible', 170, (1, 3), 0.05, (0.1, 0.3), 1.5, 0.10, 1.0),
        'M': GroupConfig('Mixed', 160, (3, 7), 0.03, (0.05, 0.2), 0.7, 0.08, 0.5),
    }
    centers = {
        'R': env_signal + np.array([2.0, -1.5, 0.5, -0.5, 1.0])[:d],
        'F': env_signal + np.array([-2.0, 1.5, -1.0, 1.0, -0.5])[:d],
        'M': env_signal + np.array([0.0, 0.0, 2.0, -1.0, -1.5])[:d],
    }

    all_agents, group_indices, offset = [], {}, 0
    for name, config in configs.items():
        agents = create_group(config, centers[name], rng)
        group_indices[name] = list(range(offset, offset + len(agents)))
        all_agents.extend(agents)
        offset += len(agents)
    N = len(all_agents)

    adj = np.zeros((N, N))
    for name, config in configs.items():
        idx = group_indices[name]
        sub = build_erdos_renyi(len(idx), config.edge_prob, rng)
        adj[np.ix_(idx, idx)] = sub
    inter = (rng.random((N, N)) < 0.005) & (adj == 0)
    inter = np.triu(inter, 1)
    adj = np.maximum(adj, (inter | inter.T).astype(float))
    np.fill_diagonal(adj, 0)

    sim = SyntheticBeliefNetwork(all_agents, adj, env_signal, seed=seed)
    if not flexibilize:
        trajectories = sim.run(n_steps, record_every=1)
        return trajectories, group_indices, env_signal

    # First half: unperturbed. Then rewrite Group R to Flexible-like
    # parameters and continue (agent beliefs carry over between runs).
    half = n_steps // 2
    traj1 = sim.run(half, record_every=1)
    for i in group_indices['R']:
        all_agents[i].obs_noise_std = 1.5
        all_agents[i].learning_rate = 0.05
        all_agents[i].precision = rng.uniform(1, 3)
        all_agents[i].social_coupling = rng.uniform(0.1, 0.3)
    traj2 = sim.run(n_steps - half, record_every=1)
    return np.concatenate([traj1, traj2], axis=0), group_indices, env_signal


# ====================================================================
# H-G1: GW separates belief manifolds without shared coordinates
# ====================================================================

def run_hg1(trajectories, group_indices, rng,
            n_per_group: int = 10, n_cloud: int = 120):
    print("\n--- H-G1: GW distance across private coordinate frames ---")
    n_steps = trajectories.shape[0]
    # Stationary, pre-perturbation window
    lo, hi = n_steps // 2 - 1500, n_steps // 2

    clouds, labels = [], []
    for name, idx in group_indices.items():
        chosen = rng.choice(idx, size=n_per_group, replace=False)
        for i in chosen:
            traj = trajectories[lo:hi, i, :]
            sub = traj[rng.choice(len(traj), size=n_cloud, replace=False)]
            Q, t = random_isometry(sub.shape[1], rng)
            clouds.append(apply_isometry(sub, Q, t))  # private frame
            labels.append(name)
    labels = np.array(labels)
    n_agents = len(clouds)

    C = [squareform(pdist(c)) for c in clouds]

    gw_mat = np.zeros((n_agents, n_agents))
    kl_mat = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            gw2, _ = entropic_gromov_wasserstein(C[i], C[j])
            gw_mat[i, j] = gw_mat[j, i] = np.sqrt(max(gw2, 0.0))
            kl = symmetrized_gaussian_kl(clouds[i], clouds[j])
            kl_mat[i, j] = kl_mat[j, i] = kl

    gw_stats = separation_stats(gw_mat, labels, rng)
    kl_stats = separation_stats(kl_mat, labels, rng)

    print(f"  GW : within={gw_stats['within_mean']:.4f}, "
          f"between={gw_stats['between_mean']:.4f}, "
          f"ratio={gw_stats['ratio']:.2f}, p={gw_stats['p_value']:.4f}")
    print(f"  KL : within={kl_stats['within_mean']:.4f}, "
          f"between={kl_stats['between_mean']:.4f}, "
          f"ratio={kl_stats['ratio']:.2f}, p={kl_stats['p_value']:.4f}")
    print("  (KL is computed in the same private frames — it should fail, "
          "GW should not.)")

    return {'gw_matrix': gw_mat, 'kl_matrix': kl_mat, 'labels': labels,
            'gw_stats': gw_stats, 'kl_stats': kl_stats}


# ====================================================================
# H-G2: Fisher geodesics — uncertainty bowing + cost of changing your mind
# ====================================================================

def run_hg2(rng):
    print("\n--- H-G2: Fisher-Rao geodesics on belief space ---")

    # Two confident beliefs, far apart in mean: z = (mu1, mu2, log sigma)
    sigma0 = 0.3
    z_a = np.array([-2.0, 0.0, np.log(sigma0)])
    z_b = np.array([2.0, 0.0, np.log(sigma0)])

    path, energy_hist = compute_geodesic(z_a, z_b, gaussian_fisher_metric,
                                         n_points=32, n_iters=400)
    straight = np.linspace(z_a, z_b, 32)

    e_straight = path_energy(straight, gaussian_fisher_metric)
    e_geo = path_energy(path, gaussian_fisher_metric)
    sigma_path = np.exp(path[:, -1])
    sigma_max = float(sigma_path.max())

    print(f"  Straight-line energy: {e_straight:.4f}")
    print(f"  Geodesic energy:      {e_geo:.4f} "
          f"({100 * (1 - e_geo / e_straight):.1f}% reduction)")
    print(f"  Peak sigma along geodesic: {sigma_max:.3f} "
          f"(endpoints: {sigma0:.3f}) — path bows through uncertainty")

    # Cost of the same mean-shift under rigid vs flexible uncertainty
    sigma_rigid, sigma_flex = 0.2, 1.5
    len_rigid = path_length(np.linspace(
        np.array([-2.0, 0.0, np.log(sigma_rigid)]),
        np.array([2.0, 0.0, np.log(sigma_rigid)]), 32), gaussian_fisher_metric)
    len_flex = path_length(np.linspace(
        np.array([-2.0, 0.0, np.log(sigma_flex)]),
        np.array([2.0, 0.0, np.log(sigma_flex)]), 32), gaussian_fisher_metric)
    print(f"  Fixed-sigma transition cost: rigid={len_rigid:.2f}, "
          f"flexible={len_flex:.2f} (ratio {len_rigid / len_flex:.1f}x)")

    # Empirical Fisher vs closed form (J. Duell's proposed reduction)
    errs = []
    for _ in range(5):
        z = np.concatenate([rng.normal(0, 2, size=2), [rng.uniform(-1.0, 0.5)]])
        obs = z[:2][None, :] + rng.normal(0, np.exp(z[-1]), size=(20000, 2))
        G_emp = empirical_fisher(gaussian_score(z, obs))
        G_true = gaussian_fisher_metric(z)
        errs.append(np.linalg.norm(G_emp - G_true) / np.linalg.norm(G_true))
    print(f"  Empirical vs closed-form Fisher: max rel. error "
          f"{max(errs):.3f} over 5 random states (n=20k samples)")

    return {'geodesic': path, 'straight': straight, 'energy_hist': energy_hist,
            'e_straight': e_straight, 'e_geodesic': e_geo,
            'sigma_max': sigma_max, 'sigma_endpoints': sigma0,
            'cost_rigid': len_rigid, 'cost_flexible': len_flex,
            'fisher_max_rel_err': float(max(errs))}


# ====================================================================
# H-G3: localizing divergence along independently-computed geodesics
# ====================================================================

def run_hg3(rng, n_path: int = 21, n_cloud: int = 80, n_repeats: int = 3):
    print("\n--- H-G3: divergence profile along matched geodesics ---")

    # Agent R (rigid, tight posteriors) and agent F (flexible, diffuse),
    # each in its own latent space (endpoint correspondence only).
    mu_start, mu_end = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    sigma_r, sigma_f = 0.3, 1.2

    paths = {}
    for name, sig in (('R', sigma_r), ('F', sigma_f)):
        z0 = np.concatenate([mu_start, [np.log(sig)]])
        z1 = np.concatenate([mu_end, [np.log(sig)]])
        paths[name], _ = compute_geodesic(z0, z1, gaussian_fisher_metric,
                                          n_points=n_path, n_iters=300)

    iso = {name: random_isometry(2, rng) for name in ('R', 'F')}

    profile = np.zeros(n_path)
    for t in range(n_path):
        vals = []
        for _ in range(n_repeats):
            clouds = {}
            for name in ('R', 'F'):
                mu, s = paths[name][t, :2], paths[name][t, -1]
                pts = mu[None, :] + rng.normal(0, np.exp(s), size=(n_cloud, 2))
                Q, tr = iso[name]
                clouds[name] = apply_isometry(pts, Q, tr)
            C1 = squareform(pdist(clouds['R']))
            C2 = squareform(pdist(clouds['F']))
            gw2, _ = entropic_gromov_wasserstein(C1, C2)
            vals.append(np.sqrt(max(gw2, 0.0)))
        profile[t] = np.median(vals)

    t_grid = np.linspace(0, 1, n_path)
    t_peak = float(t_grid[np.argmax(profile)])
    sig_gap = np.abs(np.exp(paths['R'][:, -1]) - np.exp(paths['F'][:, -1]))
    corr = float(np.corrcoef(profile, sig_gap)[0, 1])

    print(f"  Divergence profile peak at t={t_peak:.2f} "
          f"(max GW={profile.max():.4f}, min GW={profile.min():.4f})")
    print(f"  Corr(profile, |sigma_R - sigma_F| along path) = {corr:.3f}")

    return {'profile': profile, 't_grid': t_grid, 'paths': paths,
            't_peak': t_peak, 'corr_sigma_gap': corr}


# ====================================================================
# H-G4: are two groups moving toward or away from each other?
# ====================================================================

def _windowed_gw_series(trajectories, group_indices, rng, window: int,
                        n_agents_win: int, n_time_pts: int, n_repeats: int):
    """GW(R, F) per window, averaged over independent subsamples."""
    n_steps = trajectories.shape[0]
    starts = list(range(0, n_steps - window + 1, window))
    series = []
    for start in starts:
        vals = []
        for _ in range(n_repeats):
            clouds = {}
            for name in ('R', 'F'):
                idx = rng.choice(group_indices[name], size=n_agents_win,
                                 replace=False)
                tps = rng.choice(np.arange(start, start + window),
                                 size=n_time_pts, replace=False)
                pts = trajectories[np.ix_(tps, idx)].reshape(
                    -1, trajectories.shape[2])
                Q, t = random_isometry(pts.shape[1], rng)
                clouds[name] = apply_isometry(pts, Q, t)
            C1 = squareform(pdist(clouds['R']))
            C2 = squareform(pdist(clouds['F']))
            gw2, _ = entropic_gromov_wasserstein(C1, C2)
            vals.append(np.sqrt(max(gw2, 0.0)))
        series.append(np.median(vals))
    return np.array(series), np.array(starts) + window // 2


def run_hg4(traj_perturbed, traj_control, group_indices, rng,
            window: int = 400, n_agents_win: int = 30,
            n_time_pts: int = 5, n_repeats: int = 3):
    print("\n--- H-G4: sliding-window GW between Rigid and Flexible ---")
    n_steps = traj_perturbed.shape[0]
    perturb_step = n_steps // 2

    gw_pert, centers = _windowed_gw_series(
        traj_perturbed, group_indices, rng, window, n_agents_win,
        n_time_pts, n_repeats)
    gw_ctrl, _ = _windowed_gw_series(
        traj_control, group_indices, rng, window, n_agents_win,
        n_time_pts, n_repeats)

    # Skip the first window (initial transient: agents still travelling
    # from their priors to the attractor).
    pre_mask = (centers < perturb_step) & (centers > window)
    post_mask = centers > perturb_step

    results = {}
    for label, series in (('flexibilized', gw_pert), ('control', gw_ctrl)):
        pre, post = series[pre_mask], series[post_mask]
        change = (post.mean() - pre.mean()) / pre.mean()
        print(f"  [{label:13s}] GW(R,F) pre: {pre.mean():.4f} "
              f"(+/- {pre.std():.4f})  post: {post.mean():.4f} "
              f"(+/- {post.std():.4f})  change: {100 * change:+.1f}%")
        results[label] = {'pre_mean': float(pre.mean()),
                          'post_mean': float(post.mean()),
                          'relative_change': float(change)}
    # Structural noise floor: GW between two disjoint halves of the
    # Flexible group in the final window — how far apart two clouds
    # from the SAME belief-generating system come out.
    f_idx = np.array(group_indices['F'])
    half = len(f_idx) // 2
    lo = n_steps - window
    floors = []
    for _ in range(n_repeats):
        perm = rng.permutation(f_idx)
        clouds = []
        for sub in (perm[:half], perm[half:]):
            idx = rng.choice(sub, size=n_agents_win, replace=False)
            tps = rng.choice(np.arange(lo, n_steps), size=n_time_pts,
                             replace=False)
            pts = traj_perturbed[np.ix_(tps, idx)].reshape(
                -1, traj_perturbed.shape[2])
            Q, t = random_isometry(pts.shape[1], rng)
            clouds.append(apply_isometry(pts, Q, t))
        gw2, _ = entropic_gromov_wasserstein(
            squareform(pdist(clouds[0])), squareform(pdist(clouds[1])))
        floors.append(np.sqrt(max(gw2, 0.0)))
    noise_floor = float(np.median(floors))
    print(f"  Same-system noise floor GW(F half, F half): {noise_floor:.4f}")
    print(f"  (Group R made Flexible-like at t={perturb_step} in the "
          f"perturbed run; control unperturbed.)")

    return {'gw_series': gw_pert, 'gw_series_control': gw_ctrl,
            'window_centers': centers, 'perturb_step': perturb_step,
            'noise_floor': noise_floor,
            **results['flexibilized'], 'control': results['control']}


# ====================================================================
# Figures
# ====================================================================

def generate_figures(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[figures skipped — matplotlib not installed]")
        return

    from pathlib import Path
    fig_dir = Path(__file__).parent.parent / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 10, 'font.family': 'serif',
        'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 8,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    COLORS = {'R': '#c0392b', 'F': '#27ae60', 'M': '#f39c12'}
    GROUP_LABELS = {'R': 'Rigid', 'F': 'Flexible', 'M': 'Mixed'}

    # --- Figure A: GW matrix (H-G1) + within/between bars ---
    hg1 = results['hg1']
    labels = hg1['labels']
    order = np.argsort(labels)
    gw_sorted = hg1['gw_matrix'][np.ix_(order, order)]
    sorted_labels = labels[order]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2),
                             gridspec_kw={'width_ratios': [1.2, 1.0],
                                          'wspace': 0.45})
    im = axes[0].imshow(gw_sorted, cmap='viridis')
    bounds = np.where(np.diff([ord(c) for c in sorted_labels]))[0]
    for b in bounds:
        axes[0].axhline(b + 0.5, color='white', lw=1.0)
        axes[0].axvline(b + 0.5, color='white', lw=1.0)
    tick_pos, tick_lab = [], []
    for g in ('F', 'M', 'R'):
        pos = np.where(sorted_labels == g)[0]
        if len(pos):
            tick_pos.append(pos.mean())
            tick_lab.append(GROUP_LABELS[g])
    axes[0].set_xticks(tick_pos)
    axes[0].set_xticklabels(tick_lab)
    axes[0].set_yticks(tick_pos)
    axes[0].set_yticklabels(tick_lab)
    axes[0].set_title('GW distance (private frames)')
    plt.colorbar(im, ax=axes[0], fraction=0.045)

    peak = 0.0
    for k, (name, stats) in enumerate((('GW', hg1['gw_stats']),
                                       ('sym-KL', hg1['kl_stats']))):
        x = np.array([0, 1]) + k * 2.6
        axes[1].bar(x, [stats['within_mean'], stats['between_mean']],
                    color=['#7f8c8d', '#2980b9'], width=0.9)
        top = max(stats['within_mean'], stats['between_mean'])
        peak = max(peak, top)
        axes[1].text(x.mean(), top * 2.0,
                     f"{name}\np={stats['p_value']:.3f}", ha='center', fontsize=8)
    axes[1].set_xticks([0, 1, 2.6, 3.6])
    axes[1].set_xticklabels(['within', 'between', 'within', 'between'], fontsize=8)
    axes[1].set_yscale('log')
    axes[1].set_ylim(top=peak * 400)
    axes[1].set_ylabel('mean distance (log)')
    axes[1].set_title('Separation: GW vs distributional KL', pad=12)
    fig.savefig(fig_dir / 'fig_exp4_gw_matrix.pdf')
    plt.close(fig)

    # --- Figure B: Fisher geodesic (H-G2) ---
    hg2 = results['hg2']
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    geo, straight = hg2['geodesic'], hg2['straight']
    axes[0].plot(straight[:, 0], np.exp(straight[:, -1]), '--', color='#7f8c8d',
                 label='straight path')
    axes[0].plot(geo[:, 0], np.exp(geo[:, -1]), '-', color='#8e44ad', lw=2,
                 label='Fisher geodesic')
    axes[0].plot([geo[0, 0], geo[-1, 0]], [np.exp(geo[0, -1])] * 2, 'ko', ms=6)
    axes[0].set_xlabel(r'belief mean $\mu_1$')
    axes[0].set_ylabel(r'uncertainty $\sigma$')
    axes[0].set_title('Least-effort belief transition\nbows through uncertainty')
    axes[0].legend()

    axes[1].bar([0, 1], [hg2['cost_rigid'], hg2['cost_flexible']],
                color=[COLORS['R'], COLORS['F']], width=0.6)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['rigid\n' + r'($\sigma=0.2$)',
                             'flexible\n' + r'($\sigma=1.5$)'])
    axes[1].set_ylabel('Fisher path length')
    axes[1].set_title('Cost of the same belief change')
    fig.savefig(fig_dir / 'fig_exp4_geodesic.pdf')
    plt.close(fig)

    # --- Figure C: divergence profile (H-G3) + temporal GW (H-G4) ---
    hg3, hg4 = results['hg3'], results['hg4']
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    axes[0].plot(hg3['t_grid'], hg3['profile'], '-o', color='#2980b9', ms=3)
    axes[0].axvline(hg3['t_peak'], color='#c0392b', ls=':', lw=1)
    axes[0].set_xlabel('position along geodesic $t$')
    axes[0].set_ylabel('GW divergence')
    axes[0].set_title('Divergence along the transition')

    axes[1].plot(hg4['window_centers'], hg4['gw_series_control'], '-o',
                 color='#95a5a6', ms=3, label='control')
    axes[1].plot(hg4['window_centers'], hg4['gw_series'], '-o',
                 color='#2c3e50', ms=3, label='R flexibilized')
    axes[1].axvline(hg4['perturb_step'], color='#c0392b', ls=':', lw=1,
                    label='intervention')
    axes[1].axhline(hg4['noise_floor'], color='#27ae60', ls='--', lw=1,
                    label='same-system floor')
    axes[1].set_xlabel('simulation step')
    axes[1].set_ylabel('GW(Rigid, Flexible)')
    axes[1].set_title('GW(R, F) over time')
    axes[1].legend()
    fig.savefig(fig_dir / 'fig_exp4_profile.pdf')
    plt.close(fig)

    print(f"\nFigures written to {fig_dir}/fig_exp4_*.pdf")


# ====================================================================
# Main
# ====================================================================

def run_experiment(seed: int = 42, n_steps: int = 6000, d: int = 5,
                   make_figures: bool = True):
    rng = np.random.default_rng(seed)

    print("Running belief-network simulations (perturbed + control)...")
    traj_pert, group_indices, env_signal = simulate_network(
        seed=seed, n_steps=n_steps, d=d, flexibilize=True)
    traj_ctrl, _, _ = simulate_network(
        seed=seed, n_steps=n_steps, d=d, flexibilize=False)
    print(f"Trajectories shape: {traj_pert.shape}")

    results = {}
    results['hg1'] = run_hg1(traj_pert, group_indices, rng)
    results['hg2'] = run_hg2(rng)
    results['hg3'] = run_hg3(rng)
    results['hg4'] = run_hg4(traj_pert, traj_ctrl, group_indices, rng)

    if make_figures:
        generate_figures(results)

    return results


if __name__ == '__main__':
    results = run_experiment(seed=42)
    print("\n=== Experiment 4 Complete ===")
