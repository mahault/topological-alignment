"""
Experiment 1: Synthetic Belief Network
=======================================
N=500 agents with precision-weighted Bayesian updating + social coupling.
Three groups: Rigid (R), Flexible (F), Mixed (M).

Tests:
  H1: Attractor reconstruction via TDA
  H2: RDS distance vs KL divergence
  H4: Geodesic bending via precision perturbation
  H5: Empowerment-constrained vs unconstrained alignment
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy, pearsonr, permutation_test
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ====================================================================
# Agent and group definitions
# ====================================================================

@dataclass
class AgentParams:
    """Parameters for a single agent."""
    precision: float       # pi_i: inverse uncertainty
    learning_rate: float   # eta_i
    social_coupling: float # kappa_i
    obs_noise_std: float   # sigma_i
    belief: np.ndarray = field(default_factory=lambda: np.zeros(5))
    prior: np.ndarray = field(default_factory=lambda: np.zeros(5))


@dataclass
class GroupConfig:
    """Configuration for an agent group."""
    name: str
    n_agents: int
    precision_range: tuple[float, float]
    learning_rate: float
    coupling_range: tuple[float, float]
    obs_noise_std: float
    edge_prob: float
    prior_std: float  # spread of initial beliefs around group center


def create_group(config: GroupConfig, center: np.ndarray, rng: np.random.Generator) -> list[AgentParams]:
    """Create agents for a group with specified configuration."""
    agents = []
    for _ in range(config.n_agents):
        pi = rng.uniform(*config.precision_range)
        kappa = rng.uniform(*config.coupling_range)
        belief = center + rng.normal(0, config.prior_std, size=center.shape)
        agents.append(AgentParams(
            precision=pi,
            learning_rate=config.learning_rate,
            social_coupling=kappa,
            obs_noise_std=config.obs_noise_std,
            belief=belief.copy(),
            prior=belief.copy(),
        ))
    return agents


# ====================================================================
# Simulation
# ====================================================================

class SyntheticBeliefNetwork:
    """Simulate a network of belief-updating agents."""

    def __init__(
        self,
        agents: list[AgentParams],
        adjacency: np.ndarray,
        env_signal: np.ndarray,
        seed: int = 42,
    ):
        self.agents = agents
        self.N = len(agents)
        self.adj = adjacency  # (N, N) binary adjacency
        self.env_signal = env_signal  # (d,) true state
        self.d = env_signal.shape[0]
        self.rng = np.random.default_rng(seed)

        # Pre-compute neighbor lists and weights
        self.neighbors = []
        for i in range(self.N):
            nbrs = np.where(self.adj[i] > 0)[0]
            self.neighbors.append(nbrs)

        # Storage for trajectories
        self.trajectories = []  # list of (T, N, d) snapshots

    def _init_vectorized(self):
        """Pre-compute vectorized arrays for fast stepping."""
        self._beliefs = np.array([a.belief for a in self.agents])  # (N, d)
        self._precisions = np.array([a.precision for a in self.agents])  # (N,)
        self._lrs = np.array([a.learning_rate for a in self.agents])  # (N,)
        self._kappas = np.array([a.social_coupling for a in self.agents])  # (N,)
        self._noise_stds = np.array([a.obs_noise_std for a in self.agents])  # (N,)
        # Degree-normalized adjacency for social coupling
        degree = np.maximum(self.adj.sum(axis=1, keepdims=True), 1.0)
        self._adj_norm = self.adj / degree  # (N, N)

    def step(self):
        """One vectorized step of belief updating for all agents."""
        # Observations: env_signal + per-agent noise
        noise = self.rng.normal(0, 1, size=(self.N, self.d)) * self._noise_stds[:, None]
        obs = self.env_signal[None, :] + noise  # (N, d)

        # Prediction error
        pe = obs - self._beliefs  # (N, d)

        # Bayesian gain: high precision → low gain → resistant to change
        obs_prec = 1.0 / (self._noise_stds ** 2)  # (N,)
        gain = (self._lrs * obs_prec) / (self._precisions + obs_prec)  # (N,)
        update = gain[:, None] * pe  # (N, d)

        # Social coupling: adj_norm @ beliefs - beliefs = mean neighbor belief - own belief
        social = self._adj_norm @ self._beliefs - self._beliefs  # (N, d)
        social *= self._kappas[:, None]

        self._beliefs += update + social

    def run(self, n_steps: int, record_every: int = 1,
            perturbation: dict = None) -> np.ndarray:
        """Run simulation for n_steps, recording trajectories.

        perturbation: optional dict with 'step', 'agents', 'precision_factor'
            to apply mid-simulation precision scaling.
        """
        self._init_vectorized()
        T = n_steps // record_every
        trajectories = np.zeros((T, self.N, self.d))

        step_idx = 0
        for t in range(n_steps):
            self.step()
            if perturbation and t == perturbation['step']:
                for i in perturbation['agents']:
                    self._precisions[i] *= perturbation['precision_factor']
            if t % record_every == 0 and step_idx < T:
                trajectories[step_idx] = self._beliefs.copy()
                step_idx += 1

        # Sync back to agent objects
        for i, agent in enumerate(self.agents):
            agent.belief = self._beliefs[i].copy()

        self.trajectories = trajectories
        return trajectories


def build_erdos_renyi(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Build an Erdos-Renyi random graph adjacency matrix."""
    adj = (rng.random((n, n)) < p).astype(float)
    adj = np.maximum(adj, adj.T)  # symmetrize
    np.fill_diagonal(adj, 0)
    return adj


# ====================================================================
# TDA: Takens embedding + persistent homology
# ====================================================================

def takens_embedding(ts: np.ndarray, tau: int = 10, d_e: int = 10) -> np.ndarray:
    """
    Takens delay embedding of a multivariate time series.

    Parameters
    ----------
    ts : (T, d) array — time series
    tau : int — delay
    d_e : int — embedding dimension (number of delays)

    Returns
    -------
    embedded : (T - (d_e-1)*tau, d * d_e) array
    """
    T, d = ts.shape
    n_points = T - (d_e - 1) * tau
    if n_points <= 0:
        raise ValueError(f"Time series too short for tau={tau}, d_e={d_e}")

    embedded = np.zeros((n_points, d * d_e))
    for k in range(d_e):
        start = k * tau
        embedded[:, k * d:(k + 1) * d] = ts[start:start + n_points]

    return embedded


def compute_persistence(point_cloud: np.ndarray, max_dim: int = 1,
                        max_edge: float = 5.0, n_subsample: int = 500):
    """
    Compute persistent homology of a point cloud using Vietoris-Rips filtration.

    Uses giotto-tda or ripser if available, falls back to a simple
    numpy-only H0 persistence via single-linkage clustering.

    Returns persistence diagram as (n_features, 3) array: birth, death, dim.
    """
    # Subsample if needed
    if len(point_cloud) > n_subsample:
        idx = np.random.choice(len(point_cloud), n_subsample, replace=False)
        point_cloud = point_cloud[idx]

    try:
        from gtda.homology import VietorisRipsPersistence
        VR = VietorisRipsPersistence(
            homology_dimensions=list(range(max_dim + 1)),
            max_edge_length=max_edge,
        )
        diagrams = VR.fit_transform(point_cloud[np.newaxis, :, :])[0]
        return diagrams
    except (ImportError, Exception):
        pass

    # Numpy-only fallback: H0 persistence via single-linkage (union-find)
    return _h0_persistence_numpy(point_cloud, max_edge)


def _h0_persistence_numpy(points: np.ndarray, max_edge: float = 5.0) -> np.ndarray:
    """
    Compute H0 persistence diagram via single-linkage clustering.
    Pure numpy — no sklearn or ripser dependency.

    Each point starts as its own component (born at 0).
    Components merge at the pairwise distance of their closest points.
    The younger component dies at that distance.
    """
    from scipy.spatial.distance import pdist, squareform

    n = len(points)
    if n < 2:
        return np.empty((0, 3))

    # Pairwise distances (condensed form)
    dists = pdist(points)
    # Get sorted edge list
    n_edges = len(dists)
    # Convert condensed index to (i,j) pairs
    edge_dists = np.sort(dists)
    edge_order = np.argsort(dists)

    # Union-Find
    parent = list(range(n))
    rank_uf = [0] * n
    birth = [0.0] * n  # all born at epsilon=0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    diagrams = []

    for idx in range(n_edges):
        d = edge_dists[idx]
        if d > max_edge:
            break

        # Recover (i, j) from condensed index
        ci = edge_order[idx]
        # Condensed index formula: ci = n*i - i*(i+1)/2 + j - i - 1
        # Invert: i = int(n - 0.5 - sqrt((n-0.5)^2 - 2*ci))
        i = int(n - 0.5 - np.sqrt(max(0, (n - 0.5) ** 2 - 2 * ci)))
        j = int(ci - n * i + i * (i + 1) / 2 + i + 1)
        if j >= n:
            i += 1
            j = int(ci - n * i + i * (i + 1) / 2 + i + 1)

        ri, rj = find(i), find(j)
        if ri == rj:
            continue

        # Merge: younger component dies
        if rank_uf[ri] < rank_uf[rj]:
            ri, rj = rj, ri  # ri survives
        parent[rj] = ri
        if rank_uf[ri] == rank_uf[rj]:
            rank_uf[ri] += 1

        # rj dies at distance d
        if d > 1e-10:  # skip zero-persistence features
            diagrams.append((0.0, d, 0))

    return np.array(diagrams) if diagrams else np.empty((0, 3))


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray, dim: int = 0) -> float:
    """
    Approximate bottleneck distance between persistence diagrams for a given dimension.
    Uses the wasserstein-1 approximation if gudhi/persim not available.
    """
    # Filter by dimension
    if len(dgm1) > 0 and dgm1.shape[1] >= 3:
        d1 = dgm1[dgm1[:, 2] == dim][:, :2]
    else:
        d1 = np.empty((0, 2))

    if len(dgm2) > 0 and dgm2.shape[1] >= 3:
        d2 = dgm2[dgm2[:, 2] == dim][:, :2]
    else:
        d2 = np.empty((0, 2))

    # Pure numpy fallback: L-inf Wasserstein approximation
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        return np.max(d2[:, 1] - d2[:, 0]) / 2
    if len(d2) == 0:
        return np.max(d1[:, 1] - d1[:, 0]) / 2

    # Persistence values
    pers1 = d1[:, 1] - d1[:, 0]
    pers2 = d2[:, 1] - d2[:, 0]

    # Sort by persistence descending
    pers1 = np.sort(pers1)[::-1]
    pers2 = np.sort(pers2)[::-1]

    # Pad to same length
    max_len = max(len(pers1), len(pers2))
    p1 = np.zeros(max_len)
    p2 = np.zeros(max_len)
    p1[:len(pers1)] = pers1
    p2[:len(pers2)] = pers2

    return np.max(np.abs(p1 - p2))


# ====================================================================
# RDS distance estimation (Koopman operator comparison)
# ====================================================================

def estimate_koopman_dmd(trajectories: np.ndarray, rank: int = 10) -> np.ndarray:
    """
    Estimate Koopman operator via Dynamic Mode Decomposition.

    Parameters
    ----------
    trajectories : (T, d) array — time series of beliefs
    rank : int — truncation rank

    Returns
    -------
    K : (rank, rank) array — approximate Koopman matrix
    """
    X = trajectories[:-1].T  # (d, T-1)
    Y = trajectories[1:].T   # (d, T-1)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, len(S))
    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]

    # K_tilde = U_r^T Y V_r S_r^{-1}
    K_tilde = U_r.T @ Y @ Vt_r.T @ np.diag(1.0 / S_r)
    return K_tilde


def rds_distance(traj1: np.ndarray, traj2: np.ndarray, rank: int = 10) -> float:
    """
    Compute RDS misalignment distance between two belief trajectories
    via Koopman operator comparison.

    d_mis = ||K1 - K2||_op (operator norm)
    """
    K1 = estimate_koopman_dmd(traj1, rank=rank)
    K2 = estimate_koopman_dmd(traj2, rank=rank)

    # Match dimensions
    r = min(K1.shape[0], K2.shape[0])
    K1 = K1[:r, :r]
    K2 = K2[:r, :r]

    # Operator norm = largest singular value of difference
    diff = K1 - K2
    return np.linalg.svd(diff, compute_uv=False)[0]


def kl_divergence_gaussian(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Estimate KL divergence between marginal distributions of two trajectories,
    fitted as multivariate Gaussians.
    """
    mu1, cov1 = np.mean(traj1, axis=0), np.cov(traj1.T) + 1e-6 * np.eye(traj1.shape[1])
    mu2, cov2 = np.mean(traj2, axis=0), np.cov(traj2.T) + 1e-6 * np.eye(traj2.shape[1])

    d = len(mu1)
    cov2_inv = np.linalg.inv(cov2)
    diff = mu2 - mu1

    kl = 0.5 * (
        np.trace(cov2_inv @ cov1)
        + diff @ cov2_inv @ diff
        - d
        + np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    )
    return max(kl, 0.0)


# ====================================================================
# Empowerment estimation
# ====================================================================

def estimate_empowerment(agent: AgentParams, env_signal: np.ndarray,
                         n_samples: int = 1000, rng: np.random.Generator = None) -> float:
    """
    Estimate geometric empowerment: how many distinct future belief states
    can the agent reach via different perturbations?

    Empowerment = effective dimensionality of reachable set, measured as
    the entropy of the eigenvalue spectrum of the next-state covariance
    relative to a fixed-size perturbation. Flexible agents with low
    precision can explore more of belief space; rigid agents with high
    precision snap back to their attractor.

    Key: we apply perturbations to the *observation* (what the agent can
    influence), not directly to beliefs. High precision means the agent
    strongly corrects toward observations — but if it's already near
    the attractor, observations pull it BACK, reducing reachable diversity.
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(env_signal)

    # Simulate n_samples independent observation-action outcomes
    # Each "action" = choosing which direction to attend/move
    next_states = np.zeros((n_samples, d))
    for i in range(n_samples):
        # Agent gets observation from slightly different positions
        # (simulates attending to different aspects of environment)
        obs = env_signal + rng.normal(0, agent.obs_noise_std, size=d)
        pe = obs - agent.belief
        # Standard update: belief + lr * pi * PE + social (no social here)
        obs_prec = 1.0 / (agent.obs_noise_std ** 2)
        gain_i = agent.learning_rate * obs_prec / (agent.precision + obs_prec)
        next_states[i] = agent.belief + gain_i * pe

    # Empowerment = how spread out are the reachable next states?
    # Use effective rank of covariance (spectral entropy)
    cov = np.cov(next_states.T) + 1e-10 * np.eye(d)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]

    if len(eigvals) == 0:
        return 0.0

    # Normalize to probabilities
    p = eigvals / np.sum(eigvals)
    # Spectral entropy (effective dimensionality)
    spectral_entropy = -np.sum(p * np.log(p + 1e-15))

    # Also compute total variance as a scale measure
    total_var = np.sum(eigvals)

    # Empowerment = spectral_entropy * log(total_var + 1)
    # High when: many dimensions are reachable (high entropy) AND
    #            the reachable set is large (high variance)
    return spectral_entropy * np.log1p(total_var)


# ====================================================================
# Main experiment runner
# ====================================================================

def run_experiment(seed: int = 42, n_steps: int = 10000, d: int = 5):
    """Run the full Experiment 1 pipeline."""
    rng = np.random.default_rng(seed)

    # Environment: true signal
    env_signal = rng.normal(0, 1, size=d)

    # Group configurations
    configs = {
        'R': GroupConfig('Rigid', 170, (8, 12), 0.01, (0.01, 0.05), 0.5, 0.05, 0.1),
        'F': GroupConfig('Flexible', 170, (1, 3), 0.05, (0.1, 0.3), 0.5, 0.10, 1.0),
        'M': GroupConfig('Mixed', 160, (3, 7), 0.03, (0.05, 0.2), 0.5, 0.08, 0.5),
    }

    # Group centers (separated in belief space)
    centers = {
        'R': env_signal + rng.normal(0, 0.5, size=d),
        'F': env_signal + rng.normal(0, 0.5, size=d),
        'M': env_signal + rng.normal(0, 0.5, size=d),
    }

    # Create agents
    all_agents = []
    group_indices = {}
    offset = 0
    for name, config in configs.items():
        agents = create_group(config, centers[name], rng)
        group_indices[name] = list(range(offset, offset + len(agents)))
        all_agents.extend(agents)
        offset += len(agents)

    N = len(all_agents)

    # Build adjacency (within-group ER, sparse between-group)
    adj = np.zeros((N, N))
    for name, config in configs.items():
        idx = group_indices[name]
        sub_adj = build_erdos_renyi(len(idx), config.edge_prob, rng)
        for i_local, i_global in enumerate(idx):
            for j_local, j_global in enumerate(idx):
                adj[i_global, j_global] = sub_adj[i_local, j_local]

    # Sparse inter-group connections
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] == 0 and rng.random() < 0.005:
                adj[i, j] = adj[j, i] = 1.0

    # Run simulation with precision perturbation at midpoint (H4)
    print("Running simulation...")
    r_idx = group_indices['R']
    sim = SyntheticBeliefNetwork(all_agents, adj, env_signal, seed=seed)
    trajectories = sim.run(n_steps, record_every=1, perturbation={
        'step': n_steps // 2,
        'agents': r_idx,          # all of Group R
        'precision_factor': 0.25,  # quarter precision → doubles gain
    })
    print(f"Trajectories shape: {trajectories.shape}")

    results = {}

    # ---- H1: Attractor reconstruction via TDA ----
    print("\n--- H1: TDA Attractor Reconstruction ---")
    group_diagrams = {}
    for name, idx in group_indices.items():
        diagrams = []
        for i in idx[:20]:  # subsample agents for speed
            agent_traj = trajectories[:, i, :]
            embedded = takens_embedding(agent_traj, tau=10, d_e=10)
            dgm = compute_persistence(embedded, max_dim=1, n_subsample=300)
            diagrams.append(dgm)
        group_diagrams[name] = diagrams

    # Within-group vs between-group bottleneck distances
    within_dists = []
    between_dists = []
    groups = list(group_indices.keys())

    for name in groups:
        dgms = group_diagrams[name]
        for i in range(len(dgms)):
            for j in range(i + 1, len(dgms)):
                d_bn = bottleneck_distance(dgms[i], dgms[j], dim=0)
                within_dists.append(d_bn)

    for g1_idx in range(len(groups)):
        for g2_idx in range(g1_idx + 1, len(groups)):
            dgms1 = group_diagrams[groups[g1_idx]]
            dgms2 = group_diagrams[groups[g2_idx]]
            for d1 in dgms1:
                for d2 in dgms2:
                    d_bn = bottleneck_distance(d1, d2, dim=0)
                    between_dists.append(d_bn)

    within_mean = np.mean(within_dists) if within_dists else 0
    between_mean = np.mean(between_dists) if between_dists else 0
    print(f"  Within-group bottleneck: {within_mean:.4f}")
    print(f"  Between-group bottleneck: {between_mean:.4f}")
    print(f"  Ratio (between/within): {between_mean / max(within_mean, 1e-8):.2f}")
    results['h1_within'] = within_mean
    results['h1_between'] = between_mean

    # ---- H2: RDS distance vs KL ----
    print("\n--- H2: RDS Distance vs KL Divergence ---")
    # Compute group-level trajectories (centroid)
    group_centroids = {}
    for name, idx in group_indices.items():
        group_centroids[name] = np.mean(trajectories[:, idx, :], axis=1)  # (T, d)

    rds_dists = {}
    kl_dists = {}
    for g1_idx in range(len(groups)):
        for g2_idx in range(g1_idx + 1, len(groups)):
            g1, g2 = groups[g1_idx], groups[g2_idx]
            key = f"{g1}-{g2}"

            d_rds = rds_distance(group_centroids[g1], group_centroids[g2])
            d_kl = kl_divergence_gaussian(group_centroids[g1], group_centroids[g2])

            rds_dists[key] = d_rds
            kl_dists[key] = d_kl
            print(f"  {key}: RDS={d_rds:.4f}, KL={d_kl:.4f}")

    results['h2_rds'] = rds_dists
    results['h2_kl'] = kl_dists

    # ---- H4: Geodesic bending ----
    print("\n--- H4: Precision Perturbation ---")
    # Compare pre- vs post-perturbation dynamics for Group R
    # Perturbation was applied inline at t = n_steps // 2
    mid = n_steps // 2
    pre_traj = trajectories[mid - 1000:mid, r_idx, :]
    pre_direction = np.mean(np.diff(pre_traj, axis=0), axis=(0, 1))
    pre_direction /= np.linalg.norm(pre_direction) + 1e-8

    post_traj = trajectories[mid + 500:mid + 1500, r_idx, :]
    post_direction = np.mean(np.diff(post_traj, axis=0), axis=(0, 1))
    post_direction /= np.linalg.norm(post_direction) + 1e-8

    direction_change = 1.0 - np.dot(pre_direction, post_direction)
    print(f"  Direction cosine (pre vs post): {np.dot(pre_direction, post_direction):.4f}")
    print(f"  Direction change magnitude: {direction_change:.4f}")
    results['h4_direction_change'] = direction_change

    # ---- H5: Empowerment vs brittleness ----
    print("\n--- H5: Empowerment Analysis ---")
    emp_by_group = {}
    for name, idx in group_indices.items():
        emps = []
        for i in idx[:20]:
            e = estimate_empowerment(all_agents[i], env_signal, n_samples=500, rng=rng)
            emps.append(e)
        emp_by_group[name] = np.mean(emps)
        print(f"  {name}: mean empowerment = {emp_by_group[name]:.4f}")

    results['h5_empowerment'] = emp_by_group

    return results, trajectories, group_indices


if __name__ == '__main__':
    results, traj, groups = run_experiment(seed=42, n_steps=10000)
    print("\n=== Experiment 1 Complete ===")
    print(f"Results: {results}")
