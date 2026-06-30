"""
ML inference service.

Loaded once per worker process. Provides:
  - top_concepts()  : list of 20 high-degree concepts
  - score_pairs()   : rank non-neighbor pairs for a keyword
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

# torch is imported lazily inside _ensure_loaded / _load_model to keep
# import-time cost low and allow running tests without torch installed.
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)

# ── module-level singletons (cached per process) ─────────────────────────────
_graph: Optional[nx.Graph] = None
_article_embeddings: Optional[Dict] = None
_concept_embeddings: Optional[Dict[str, np.ndarray]] = None
_model: Optional[object] = None  # torch.nn.Module when loaded
_scaler: Optional[StandardScaler] = None
_top_concepts: Optional[List[str]] = None
_node_to_idx: Optional[Dict[str, int]] = None


# ── MLP model definition (mirrors original) ──────────────────────────────────
# Defined at import time only when torch is available.
if _TORCH_AVAILABLE:
    class LinkPredictionMLP(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [
                    torch.nn.Linear(prev, h),
                    torch.nn.BatchNorm1d(h),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                ]
                prev = h
            layers.append(torch.nn.Linear(prev, 1))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


def _load_model(path: str):
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[name-defined]
    model = LinkPredictionMLP(ckpt["input_dim"], ckpt["hidden_dims"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ── graph feature helpers ─────────────────────────────────────────────────────

def _nbrs(graph: nx.Graph, node: str) -> Set[str]:
    return set(graph.neighbors(node)) if graph.has_node(node) else set()


def _compute_graph_features(
    pairs: List[Tuple[str, str]], graph: nx.Graph
) -> Tuple[np.ndarray, List[str]]:
    """Compute 16 basic graph topology features per pair."""
    neighbor_cache: Dict[str, Set[str]] = {}

    def nbrs(c: str) -> Set[str]:
        if c not in neighbor_cache:
            neighbor_cache[c] = _nbrs(graph, c)
        return neighbor_cache[c]

    n = len(pairs)
    parts: List[np.ndarray] = []
    names: List[str] = []

    def add(vals, name):
        parts.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
        names.append(name)

    # co-occurrence count
    add([float(graph[a][b]["weight"]) if graph.has_edge(a, b) else 0.0 for a, b in pairs], "cooc_count")
    # common neighbours
    add([len(nbrs(a) & nbrs(b)) for a, b in pairs], "common_neighbors")
    # jaccard
    add([len(nbrs(a) & nbrs(b)) / max(len(nbrs(a) | nbrs(b)), 1) for a, b in pairs], "jaccard")
    # adamic-adar
    aa_vals = []
    for a, b in pairs:
        common = nbrs(a) & nbrs(b)
        aa_vals.append(sum(1.0 / np.log(max(len(nbrs(z)), 2)) for z in common))
    add(aa_vals, "adamic_adar")
    # preferential attachment
    add([len(nbrs(a)) * len(nbrs(b)) for a, b in pairs], "pref_attachment")
    # resource allocation
    ra_vals = []
    for a, b in pairs:
        common = nbrs(a) & nbrs(b)
        ra_vals.append(sum(1.0 / max(len(nbrs(z)), 1) for z in common))
    add(ra_vals, "resource_allocation")
    # salton
    add([len(nbrs(a) & nbrs(b)) / max(np.sqrt(len(nbrs(a)) * len(nbrs(b))), 1e-10) for a, b in pairs], "salton")
    # hub promoted
    add([len(nbrs(a) & nbrs(b)) / max(min(len(nbrs(a)), len(nbrs(b))), 1) for a, b in pairs], "hub_promoted")
    # hub suppressed
    add([len(nbrs(a) & nbrs(b)) / max(max(len(nbrs(a)), len(nbrs(b))), 1) for a, b in pairs], "hub_suppressed")
    # concept frequency (5 features)
    freq_a = np.array([float(graph.nodes[a]["freq"]) if graph.has_node(a) else 0.0 for a, _ in pairs], dtype=np.float32)
    freq_b = np.array([float(graph.nodes[b]["freq"]) if graph.has_node(b) else 0.0 for _, b in pairs], dtype=np.float32)
    parts += [
        freq_a.reshape(-1, 1), freq_b.reshape(-1, 1),
        np.minimum(freq_a, freq_b).reshape(-1, 1),
        np.maximum(freq_a, freq_b).reshape(-1, 1),
        (freq_a + freq_b).reshape(-1, 1),
    ]
    names += ["freq_a", "freq_b", "freq_min", "freq_max", "freq_sum"]
    # year trend (2 features)
    for label_idx, pos_label in [(0, "a"), (1, "b")]:
        vals = []
        for pair in pairs:
            c = pair[label_idx]
            if graph.has_node(c):
                yf = graph.nodes[c].get("year_freq", {})
                if len(yf) >= 2:
                    ys = sorted(yf.keys())
                    freqs = [yf[y] for y in ys]
                    slope = float(np.polyfit(np.arange(len(freqs), dtype=float), freqs, 1)[0])
                else:
                    slope = 0.0
            else:
                slope = 0.0
            vals.append(slope)
        parts.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
        names.append(f"trend_{pos_label}")

    return np.hstack(parts), names


def _build_sparse_adj(
    graph: nx.Graph, node_to_idx: Dict[str, int], n_nodes: int, year: int
) -> sparse.csr_matrix:
    rows, cols, vals = [], [], []
    for u, v, data in graph.edges(data=True):
        if u not in node_to_idx or v not in node_to_idx:
            continue
        w = float(data.get("year_weights", {}).get(year, 0))
        if w == 0:
            continue
        i, j = node_to_idx[u], node_to_idx[v]
        rows += [i, j]; cols += [j, i]; vals += [w, w]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)


def _compute_sparse_features(
    pairs: List[Tuple[str, str]],
    graph: nx.Graph,
    sparse_years: List[int],
    node_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute 5 per-year sparse matrix features (deg, deg2, aa_score).
    Avoids full A².toarray() by using sparse row-vector operations.
    """
    n_nodes = len(node_to_idx)
    n_pairs = len(pairs)
    v1_idx = np.array([node_to_idx.get(a, 0) for a, _ in pairs], dtype=np.int32)
    v2_idx = np.array([node_to_idx.get(b, 0) for _, b in pairs], dtype=np.int32)

    features = np.zeros((n_pairs, 5 * len(sparse_years)), dtype=np.float32)
    names: List[str] = []

    for i, year in enumerate(sorted(sparse_years)):
        A = _build_sparse_adj(graph, node_to_idx, n_nodes, year)

        # degree: row sums
        deg = np.array(A.sum(axis=1)).ravel()
        max_deg = max(float(deg.max()), 1.0)
        deg_norm = deg / max_deg

        # deg2: row sums of A² = A @ deg (one sparse matrix-vector multiply)
        deg2 = A.dot(deg)
        # max of A² diagonal = max squared L2 norm of rows
        a_sq_row_norms = np.array(A.power(2).sum(axis=1)).ravel()
        max_a2 = max(float(a_sq_row_norms.max()), 1.0)
        deg2_norm = deg2 / max_a2

        # aa_score[k] = A²[v1_idx[k], v2_idx[k]] / max_a2
        # Efficient: for each unique v1, compute A[v1] @ A (one row × full matrix)
        unique_v1 = np.unique(v1_idx)
        a2_row: Dict[int, np.ndarray] = {}
        for u in unique_v1:
            # A[u, :] is a sparse row vector; multiply by A gives 1×n dense result
            row = A.getrow(u)
            a2_row[u] = np.array(row.dot(A).todense()).ravel()

        aa_vals = np.array([a2_row[v1_idx[k]][v2_idx[k]] / max_a2 for k in range(n_pairs)], dtype=np.float32)

        base = i * 5
        features[:, base + 0] = deg_norm[v1_idx]
        features[:, base + 1] = deg_norm[v2_idx]
        features[:, base + 2] = deg2_norm[v1_idx]
        features[:, base + 3] = deg2_norm[v2_idx]
        features[:, base + 4] = aa_vals

        names += [f"deg_y{i}_a", f"deg_y{i}_b", f"deg2_y{i}_a", f"deg2_y{i}_b", f"aa_score_y{i}"]

    return features, names


def _compute_embedding_features(
    pairs: List[Tuple[str, str]], concept_embeddings: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, List[str]]:
    """Compute cosine + L2 distance between concept embedding pairs."""
    zero = np.zeros(next(iter(concept_embeddings.values())).shape[0], dtype=np.float32)
    emb_a = np.stack([concept_embeddings.get(a, zero) for a, _ in pairs]).astype(np.float32)
    emb_b = np.stack([concept_embeddings.get(b, zero) for _, b in pairs]).astype(np.float32)

    na = np.linalg.norm(emb_a, axis=1, keepdims=True).clip(1e-10)
    nb = np.linalg.norm(emb_b, axis=1, keepdims=True).clip(1e-10)
    cosine = np.sum((emb_a / na) * (emb_b / nb), axis=1, keepdims=True)
    l2 = np.linalg.norm(emb_a - emb_b, axis=1, keepdims=True)

    return np.hstack([cosine, l2]).astype(np.float32), ["emb_cosine", "emb_l2"]


def _compute_all_features(
    pairs: List[Tuple[str, str]],
    graph: nx.Graph,
    concept_embeddings: Dict[str, np.ndarray],
    sparse_years: List[int],
    node_to_idx: Dict[str, int],
) -> np.ndarray:
    gf, _ = _compute_graph_features(pairs, graph)
    sf, _ = _compute_sparse_features(pairs, graph, sparse_years, node_to_idx)
    ef, _ = _compute_embedding_features(pairs, concept_embeddings)
    return np.hstack([gf, sf, ef]).astype(np.float32)


def _compute_concept_embeddings(
    graph: nx.Graph, article_embeddings: Dict
) -> Dict[str, np.ndarray]:
    from collections import defaultdict
    concept_article_ids: Dict[str, Set] = defaultdict(set)
    for u, v, data in graph.edges(data=True):
        for c in (u, v):
            concept_article_ids[c].update(data.get("articles", []))

    result: Dict[str, np.ndarray] = {}
    for concept, article_ids in concept_article_ids.items():
        embs = [article_embeddings[aid] for aid in article_ids if aid in article_embeddings]
        if not embs:
            continue
        mean_emb = np.mean(embs, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm
        result[concept] = mean_emb
    return result


# ── public interface ──────────────────────────────────────────────────────────

def _ensure_loaded() -> None:
    global _graph, _article_embeddings, _concept_embeddings, _model, _scaler, _top_concepts, _node_to_idx

    if _graph is not None:
        return

    data_dir = Path(settings.data_dir)

    logger.info("Loading graph...")
    with open(data_dir / settings.graph_filename, "rb") as f:
        _graph = pickle.load(f)
    logger.info("Graph loaded: %d nodes, %d edges", _graph.number_of_nodes(), _graph.number_of_edges())

    logger.info("Loading article embeddings...")
    with open(data_dir / settings.embeddings_filename, "rb") as f:
        _article_embeddings = pickle.load(f)

    logger.info("Computing concept embeddings...")
    _concept_embeddings = _compute_concept_embeddings(_graph, _article_embeddings)
    logger.info("Concept embeddings: %d concepts", len(_concept_embeddings))

    logger.info("Loading MLP model...")
    _model = _load_model(str(data_dir / settings.model_filename))

    # top-20 concepts by degree
    degrees = sorted(_graph.degree(), key=lambda x: x[1], reverse=True)
    _top_concepts = [c for c, _ in degrees[: settings.top_k_concepts]]

    # node index map for sparse features
    _node_to_idx = {n: i for i, n in enumerate(sorted(_graph.nodes()))}

    # fit scaler on a sample of pairs
    logger.info("Fitting scaler on graph sample...")
    rng = np.random.default_rng(42)
    all_nodes = list(_graph.nodes())
    n_sample = min(3000, len(all_nodes) * (len(all_nodes) - 1) // 2)
    sample_pairs: List[Tuple[str, str]] = []
    # take all edges first
    edges = list(_graph.edges())
    rng.shuffle(edges)
    sample_pairs += [(a, b) for a, b in edges[: n_sample // 2]]
    # fill rest with random non-edges
    edge_set = {(min(a, b), max(a, b)) for a, b in _graph.edges()}
    attempts = 0
    while len(sample_pairs) < n_sample and attempts < n_sample * 20:
        i, j = rng.integers(0, len(all_nodes), size=2)
        if i != j:
            a, b = all_nodes[i], all_nodes[j]
            pair = (min(a, b), max(a, b))
            if pair not in edge_set:
                sample_pairs.append(pair)
        attempts += 1

    X_sample = _compute_all_features(
        sample_pairs, _graph, _concept_embeddings, settings.sparse_years, _node_to_idx
    )
    _scaler = StandardScaler()
    _scaler.fit(X_sample)
    # don't rescale cosine (already in [-1,1])
    n_feat = X_sample.shape[1]
    if n_feat >= 2:
        _scaler.mean_[-1] = 0.0
        _scaler.scale_[-1] = 1.0
        _scaler.mean_[-2] = 0.0
        _scaler.scale_[-2] = 1.0
    logger.info("MLService ready. Input dim=%d", n_feat)


def get_top_concepts() -> List[str]:
    _ensure_loaded()
    return list(_top_concepts)  # type: ignore[arg-type]


def _sample_candidates(
    keyword: str,
    graph: nx.Graph,
    neighbors: Set[str],
    n_total: int,
    strategy_weights: dict,
    rng: np.random.Generator,
) -> List[str]:
    """
    Sample up to `n_total` non-neighbor nodes for `keyword` using a mix of:
      2hop         – 2-hop nodes weighted by common-neighbour count (hard negatives)
      resource_alloc – 2-hop nodes weighted by Σ 1/deg(common_nb)
      pref_attach  – all nodes weighted by degree (popular concepts)
      random       – uniform random

    Returns a deduplicated list of candidate concept names.
    """
    all_nodes = np.array([n for n in graph.nodes() if n not in neighbors])
    if len(all_nodes) == 0:
        return []

    kw_nb = set(graph.neighbors(keyword))
    all_nb: Dict[str, Set[str]] = {}  # cached neighbor sets

    def get_nb(n: str) -> Set[str]:
        if n not in all_nb:
            all_nb[n] = set(graph.neighbors(n))
        return all_nb[n]

    seen: Set[str] = set()
    result: List[str] = []

    def add(node: str) -> bool:
        if node not in seen and node not in neighbors:
            seen.add(node)
            result.append(node)
            return True
        return False

    # ── 2-hop: nodes sharing at least one common neighbour with keyword ──
    n_2hop = int(n_total * strategy_weights.get("2hop", 0.55))
    pool_2hop: Dict[str, int] = {}
    for nb in kw_nb:
        for two_hop in get_nb(nb):
            if two_hop not in neighbors:
                pool_2hop[two_hop] = pool_2hop.get(two_hop, 0) + 1
    if pool_2hop and n_2hop > 0:
        pl = np.array(list(pool_2hop.keys()))
        pw = np.array([pool_2hop[x] for x in pl], dtype=np.float64)
        pw /= pw.sum()
        chosen = rng.choice(len(pl), size=min(n_2hop, len(pl)), replace=False, p=pw)
        for i in chosen:
            add(pl[i])

    # ── resource allocation: 2-hop weighted by Σ 1/deg(common_nb) ──
    n_ra = int(n_total * strategy_weights.get("resource_alloc", 0.30))
    pool_ra: Dict[str, float] = {}
    for nb in kw_nb:
        d_nb = max(graph.degree(nb), 1)
        for two_hop in get_nb(nb):
            if two_hop not in neighbors:
                pool_ra[two_hop] = pool_ra.get(two_hop, 0.0) + 1.0 / d_nb
    if pool_ra and n_ra > 0:
        pl = np.array(list(pool_ra.keys()))
        pw = np.array([pool_ra[x] for x in pl], dtype=np.float64)
        pw /= pw.sum()
        chosen = rng.choice(len(pl), size=min(n_ra, len(pl)), replace=False, p=pw)
        for i in chosen:
            add(pl[i])

    # ── preferential attachment: weighted by degree ──
    n_pa = int(n_total * strategy_weights.get("pref_attach", 0.10))
    if n_pa > 0 and len(all_nodes) > 0:
        deg_w = np.array([float(graph.degree(n)) for n in all_nodes], dtype=np.float64)
        deg_w = deg_w / deg_w.sum() if deg_w.sum() > 0 else np.ones(len(all_nodes)) / len(all_nodes)
        pa_chosen = rng.choice(len(all_nodes), size=min(n_pa * 3, len(all_nodes)), replace=False, p=deg_w)
        count = 0
        for i in pa_chosen:
            if add(all_nodes[i]):
                count += 1
                if count >= n_pa:
                    break

    # ── random: fill remaining quota ──
    n_rand = n_total - len(result)
    if n_rand > 0 and len(all_nodes) > 0:
        rand_chosen = rng.choice(len(all_nodes), size=min(n_rand * 3, len(all_nodes)), replace=False)
        count = 0
        for i in rand_chosen:
            if add(all_nodes[i]):
                count += 1
                if count >= n_rand:
                    break

    logger.info(
        "Candidate sampling for '%s': 2hop=%d ra=%d pa=%d rand=%d → total=%d",
        keyword, n_2hop, n_ra, n_pa, n_rand, len(result),
    )
    return result


def score_pairs_for_keyword(keyword: str) -> List[dict]:
    """
    Sample candidate non-neighbor nodes for `keyword` using biased strategies,
    compute 43 features, score with MLP, return top-k results.
    """
    _ensure_loaded()
    graph = _graph
    assert graph is not None

    if not graph.has_node(keyword):
        raise ValueError(f"Keyword '{keyword}' not found in graph")

    neighbors: Set[str] = set(graph.neighbors(keyword)) | {keyword}

    rng = np.random.default_rng(42)
    candidate_nodes = _sample_candidates(
        keyword, graph, neighbors,
        n_total=settings.n_candidates,
        strategy_weights=settings.candidate_strategy_weights,
        rng=rng,
    )

    if not candidate_nodes:
        return []

    pairs = [(min(keyword, c), max(keyword, c)) for c in candidate_nodes]

    X = _compute_all_features(pairs, graph, _concept_embeddings, settings.sparse_years, _node_to_idx)  # type: ignore[arg-type]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = _scaler.transform(X).astype(np.float32)  # type: ignore[union-attr]

    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed — cannot run inference")
    with torch.no_grad():  # type: ignore[name-defined]
        tensor = torch.from_numpy(X_scaled)  # type: ignore[name-defined]
        scores = torch.sigmoid(_model(tensor)).numpy()  # type: ignore[name-defined,misc]

    top_idx = np.argsort(scores)[::-1][: settings.top_k_results]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        a, b = pairs[idx]
        results.append({"concept_a": a, "concept_b": b, "score": float(scores[idx]), "rank": rank})
    return results
