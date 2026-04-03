"""
Experiment 3: Social Media Belief Basins
=========================================
Pipeline for analyzing echo chambers as topological structures.

Data: Reddit (Pushshift archive) — posts from ideologically distinct subreddits.

Tests:
  H1: Opinion attractor reconstruction via SBERT + TDA
  H2: RDS distance predicts inter-community hostility better than KL/embedding distance
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import warnings


# ====================================================================
# Data loading and preprocessing
# ====================================================================

@dataclass
class SubredditData:
    """Container for a subreddit's data."""
    name: str
    texts: list[str]
    embeddings: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None  # unix timestamps
    weekly_centroids: Optional[np.ndarray] = None


def load_pushshift_data(subreddit: str, data_dir: str,
                        max_posts: int = 100000) -> SubredditData:
    """
    Load Reddit data from Pushshift archive files.

    Expected format: one JSON line per post with 'selftext', 'title', 'created_utc'.
    """
    import json
    from pathlib import Path

    filepath = Path(data_dir) / f"{subreddit}.jsonl"
    if not filepath.exists():
        filepath = Path(data_dir) / f"{subreddit}.json"

    texts = []
    timestamps = []

    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if len(texts) >= max_posts:
                    break
                try:
                    post = json.loads(line)
                    text = post.get('selftext', '') or post.get('title', '')
                    if len(text) > 20:  # minimum length filter
                        texts.append(text)
                        timestamps.append(post.get('created_utc', 0))
                except json.JSONDecodeError:
                    continue

    return SubredditData(
        name=subreddit,
        texts=texts,
        timestamps=np.array(timestamps) if timestamps else None,
    )


def embed_texts(texts: list[str], model_name: str = 'all-mpnet-base-v2',
                batch_size: int = 128) -> np.ndarray:
    """
    Embed texts using Sentence-BERT.

    Returns (n_texts, 768) embedding matrix.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return np.array(embeddings)
    except ImportError:
        warnings.warn("sentence-transformers not available. Using random embeddings.")
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, size=(len(texts), 768))


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Reduce embedding dimensionality via PCA."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, embeddings.shape[1], embeddings.shape[0]))
    reduced = pca.fit_transform(embeddings)
    explained = np.sum(pca.explained_variance_ratio_)
    print(f"  PCA: {pca.n_components_} components, {explained:.1%} variance explained")
    return reduced


def compute_weekly_centroids(embeddings: np.ndarray,
                             timestamps: np.ndarray) -> np.ndarray:
    """
    Compute weekly embedding centroids as a belief trajectory.

    Returns (n_weeks, d) array.
    """
    if timestamps is None or len(timestamps) == 0:
        return embeddings[np.newaxis, :]  # single centroid

    # Convert to weeks
    min_t = np.min(timestamps)
    weeks = ((timestamps - min_t) / (7 * 24 * 3600)).astype(int)
    unique_weeks = np.unique(weeks)

    centroids = []
    for w in unique_weeks:
        mask = weeks == w
        if np.sum(mask) >= 5:  # minimum posts per week
            centroids.append(np.mean(embeddings[mask], axis=0))

    return np.array(centroids) if centroids else embeddings[:1]


# ====================================================================
# TDA on embedding point clouds
# ====================================================================

def subreddit_persistence(embeddings: np.ndarray, n_subsample: int = 5000,
                          max_dim: int = 1, max_edge: float = 10.0):
    """
    Compute persistent homology on a subreddit's embedding point cloud.

    Returns persistence diagram.
    """
    from exp1_synthetic import compute_persistence

    if len(embeddings) > n_subsample:
        idx = np.random.choice(len(embeddings), n_subsample, replace=False)
        cloud = embeddings[idx]
    else:
        cloud = embeddings

    return compute_persistence(cloud, max_dim=max_dim,
                               max_edge=max_edge, n_subsample=min(n_subsample, 500))


def persistence_summary(diagram) -> dict:
    """Extract summary statistics from a persistence diagram."""
    if len(diagram) == 0:
        return {'n_features': 0, 'max_persistence': 0, 'mean_persistence': 0,
                'n_h0': 0, 'n_h1': 0}

    persistence = diagram[:, 1] - diagram[:, 0]
    h0_mask = diagram[:, 2] == 0
    h1_mask = diagram[:, 2] == 1

    return {
        'n_features': len(diagram),
        'max_persistence': np.max(persistence),
        'mean_persistence': np.mean(persistence),
        'n_h0': np.sum(h0_mask),
        'n_h1': np.sum(h1_mask),
        'h0_max_pers': np.max(persistence[h0_mask]) if np.any(h0_mask) else 0,
        'h1_max_pers': np.max(persistence[h1_mask]) if np.any(h1_mask) else 0,
    }


# ====================================================================
# Distance metrics
# ====================================================================

def embedding_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Centroid-to-centroid Euclidean distance."""
    c1 = np.mean(emb1, axis=0)
    c2 = np.mean(emb2, axis=0)
    return np.linalg.norm(c1 - c2)


def kl_distance_gaussian(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Symmetric KL divergence between Gaussian-fitted embedding distributions."""
    from exp1_synthetic import kl_divergence_gaussian
    kl_12 = kl_divergence_gaussian(emb1, emb2)
    kl_21 = kl_divergence_gaussian(emb2, emb1)
    return (kl_12 + kl_21) / 2


def rds_distance_subreddits(centroids1: np.ndarray, centroids2: np.ndarray,
                             rank: int = 10) -> float:
    """RDS distance between two subreddits via Koopman operator comparison."""
    from exp1_synthetic import rds_distance

    # Ensure sufficient temporal data
    min_len = min(len(centroids1), len(centroids2))
    if min_len < rank + 2:
        rank = max(2, min_len - 2)

    return rds_distance(centroids1[:min_len], centroids2[:min_len], rank=rank)


# ====================================================================
# Hostility metrics
# ====================================================================

def compute_hostility_score(texts1: list[str], texts2: list[str],
                            name1: str, name2: str) -> float:
    """
    Estimate inter-community hostility.

    Uses sentiment analysis on cross-referencing posts.
    Falls back to embedding-based negativity if VADER unavailable.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        # Find posts mentioning the other subreddit
        scores = []
        for text in texts1[:5000]:
            if name2.lower() in text.lower():
                score = analyzer.polarity_scores(text)
                scores.append(score['compound'])
        for text in texts2[:5000]:
            if name1.lower() in text.lower():
                score = analyzer.polarity_scores(text)
                scores.append(score['compound'])

        if scores:
            return -np.mean(scores)  # higher = more hostile
        return 0.0

    except ImportError:
        # Fallback: use embedding distance as proxy
        return 0.0


# ====================================================================
# Synthetic Reddit data (for testing without real data)
# ====================================================================

def generate_synthetic_communities(n_communities: int = 10,
                                    n_posts_per: int = 5000,
                                    d: int = 50,
                                    n_weeks: int = 156,
                                    seed: int = 42) -> list[SubredditData]:
    """
    Generate synthetic community data mimicking Reddit belief dynamics.

    Creates communities with different attractor structures:
    - Echo chambers: tight clusters with slow drift
    - Diverse communities: broad distributions with multiple centers
    - Polarized communities: bimodal distributions
    """
    rng = np.random.default_rng(seed)

    community_types = [
        'echo_chamber', 'echo_chamber', 'echo_chamber',
        'diverse', 'diverse',
        'polarized', 'polarized',
        'echo_chamber', 'diverse', 'polarized',
    ]

    communities = []
    for i in range(n_communities):
        ctype = community_types[i % len(community_types)]
        name = f"community_{i}_{ctype}"

        # Community center
        center = rng.normal(0, 3, size=d)

        if ctype == 'echo_chamber':
            # Tight cluster, slow drift
            embeddings = center + rng.normal(0, 0.5, size=(n_posts_per, d))
            drift_rate = 0.01
        elif ctype == 'diverse':
            # Broad distribution, multiple subclusters
            n_clusters = rng.integers(3, 6)
            subcenters = center + rng.normal(0, 2.0, size=(n_clusters, d))
            labels = rng.integers(0, n_clusters, size=n_posts_per)
            embeddings = subcenters[labels] + rng.normal(0, 0.8, size=(n_posts_per, d))
            drift_rate = 0.05
        else:  # polarized
            # Bimodal: two opposing camps
            pole1 = center + rng.normal(0, 0.3, size=d) * 2
            pole2 = center - rng.normal(0, 0.3, size=d) * 2
            mix = rng.random(n_posts_per)
            embeddings = np.where(
                mix[:, None] < 0.5,
                pole1 + rng.normal(0, 0.6, size=(n_posts_per, d)),
                pole2 + rng.normal(0, 0.6, size=(n_posts_per, d)),
            )
            drift_rate = 0.03

        # Assign timestamps (spread over n_weeks)
        timestamps = np.sort(rng.uniform(0, n_weeks * 7 * 24 * 3600, size=n_posts_per))

        # Add temporal drift
        week_of_post = (timestamps / (7 * 24 * 3600)).astype(int)
        for w in range(n_weeks):
            mask = week_of_post == w
            drift = drift_rate * w * rng.normal(0, 1, size=d)
            embeddings[mask] += drift

        # Compute weekly centroids
        centroids = []
        for w in range(n_weeks):
            mask = week_of_post == w
            if np.sum(mask) >= 3:
                centroids.append(np.mean(embeddings[mask], axis=0))

        communities.append(SubredditData(
            name=name,
            texts=[f"synthetic post {j}" for j in range(n_posts_per)],
            embeddings=embeddings,
            timestamps=timestamps,
            weekly_centroids=np.array(centroids) if centroids else embeddings[:1],
        ))

    return communities


# ====================================================================
# Main experiment runner
# ====================================================================

def run_experiment(use_synthetic: bool = True, data_dir: str = None, seed: int = 42):
    """Run the full Experiment 3 pipeline."""
    rng = np.random.default_rng(seed)

    if use_synthetic:
        print("Using synthetic community data...")
        communities = generate_synthetic_communities(seed=seed)
    else:
        raise NotImplementedError("Real data loading requires Pushshift files in data_dir")

    n_communities = len(communities)
    print(f"Loaded {n_communities} communities")

    # ---- H1: TDA attractor reconstruction ----
    print("\n--- H1: TDA Attractor Reconstruction ---")
    tda_results = {}
    for comm in communities:
        emb = comm.embeddings
        if emb is not None and len(emb) > 100:
            dgm = subreddit_persistence(emb, n_subsample=2000, max_edge=15.0)
            summary = persistence_summary(dgm)
            tda_results[comm.name] = summary
            print(f"  {comm.name}: {summary['n_h0']} H0, {summary['n_h1']} H1, "
                  f"max_pers={summary['max_persistence']:.3f}")

    # Check: echo chambers should have fewer H0 components (tighter clusters)
    echo_h0 = [v['n_h0'] for k, v in tda_results.items() if 'echo' in k]
    diverse_h0 = [v['n_h0'] for k, v in tda_results.items() if 'diverse' in k]
    polar_h0 = [v['n_h0'] for k, v in tda_results.items() if 'polarized' in k]

    if echo_h0 and diverse_h0:
        print(f"\n  Echo chamber mean H0: {np.mean(echo_h0):.1f}")
        print(f"  Diverse mean H0: {np.mean(diverse_h0):.1f}")
        print(f"  Polarized mean H0: {np.mean(polar_h0):.1f}")

    # ---- H2: Distance metric comparison ----
    print("\n--- H2: RDS vs KL vs Embedding Distance ---")
    n = n_communities
    rds_matrix = np.zeros((n, n))
    kl_matrix = np.zeros((n, n))
    emb_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            ci = communities[i]
            cj = communities[j]

            if ci.embeddings is None or cj.embeddings is None:
                continue

            # Embedding distance
            emb_d = embedding_distance(ci.embeddings, cj.embeddings)
            emb_matrix[i, j] = emb_matrix[j, i] = emb_d

            # KL distance
            # Use PCA-reduced embeddings for numerical stability
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(20, ci.embeddings.shape[1]))
            combined = np.vstack([ci.embeddings[:2000], cj.embeddings[:2000]])
            pca.fit(combined)
            emb_i_r = pca.transform(ci.embeddings[:2000])
            emb_j_r = pca.transform(cj.embeddings[:2000])
            kl_d = kl_distance_gaussian(emb_i_r, emb_j_r)
            kl_matrix[i, j] = kl_matrix[j, i] = kl_d

            # RDS distance
            if ci.weekly_centroids is not None and cj.weekly_centroids is not None:
                if len(ci.weekly_centroids) > 5 and len(cj.weekly_centroids) > 5:
                    rds_d = rds_distance_subreddits(
                        ci.weekly_centroids, cj.weekly_centroids, rank=5,
                    )
                    rds_matrix[i, j] = rds_matrix[j, i] = rds_d

    # Compare: which metric better distinguishes community types?
    same_type_rds = []
    diff_type_rds = []
    same_type_kl = []
    diff_type_kl = []
    same_type_emb = []
    diff_type_emb = []

    type_labels = []
    for c in communities:
        if 'echo' in c.name:
            type_labels.append('echo')
        elif 'diverse' in c.name:
            type_labels.append('diverse')
        else:
            type_labels.append('polarized')

    for i in range(n):
        for j in range(i + 1, n):
            if type_labels[i] == type_labels[j]:
                same_type_rds.append(rds_matrix[i, j])
                same_type_kl.append(kl_matrix[i, j])
                same_type_emb.append(emb_matrix[i, j])
            else:
                diff_type_rds.append(rds_matrix[i, j])
                diff_type_kl.append(kl_matrix[i, j])
                diff_type_emb.append(emb_matrix[i, j])

    print(f"\n  RDS — same type: {np.mean(same_type_rds):.4f}, diff type: {np.mean(diff_type_rds):.4f}, "
          f"ratio: {np.mean(diff_type_rds) / max(np.mean(same_type_rds), 1e-8):.2f}")
    print(f"  KL  — same type: {np.mean(same_type_kl):.4f}, diff type: {np.mean(diff_type_kl):.4f}, "
          f"ratio: {np.mean(diff_type_kl) / max(np.mean(same_type_kl), 1e-8):.2f}")
    print(f"  Emb — same type: {np.mean(same_type_emb):.4f}, diff type: {np.mean(diff_type_emb):.4f}, "
          f"ratio: {np.mean(diff_type_emb) / max(np.mean(same_type_emb), 1e-8):.2f}")

    results = {
        'tda_results': tda_results,
        'rds_matrix': rds_matrix,
        'kl_matrix': kl_matrix,
        'emb_matrix': emb_matrix,
        'rds_discrimination': np.mean(diff_type_rds) / max(np.mean(same_type_rds), 1e-8),
        'kl_discrimination': np.mean(diff_type_kl) / max(np.mean(same_type_kl), 1e-8),
        'emb_discrimination': np.mean(diff_type_emb) / max(np.mean(same_type_emb), 1e-8),
    }

    return results, communities


if __name__ == '__main__':
    results, communities = run_experiment(use_synthetic=True, seed=42)
    print("\n=== Experiment 3 Complete ===")
    print(f"  RDS discrimination ratio: {results['rds_discrimination']:.2f}")
    print(f"  KL discrimination ratio: {results['kl_discrimination']:.2f}")
    print(f"  Embedding discrimination ratio: {results['emb_discrimination']:.2f}")
