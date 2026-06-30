"""Tests for ML and LLM services (unit-level, no real model/graph needed)."""

import numpy as np
import pytest


# ── LLM service tests ─────────────────────────────────────────────────────────

def test_generate_draft_no_api_key(monkeypatch):
    """Without a Groq API key the template draft is returned."""
    monkeypatch.setattr("app.config.settings.groq_api_key", "")
    from app.services.llm_service import generate_draft

    result = generate_draft("deep learning", "graph neural network")
    assert "deep learning" in result
    assert "graph neural network" in result
    assert "# Research Draft" in result


def test_generate_draft_with_api_key(monkeypatch):
    """With a Groq API key the Groq client is called."""
    monkeypatch.setattr("app.config.settings.groq_api_key", "fake-key")

    mock_content = "# AI-Generated Draft\n\n## Section 1\n..."
    mock_choice = type("Choice", (), {"message": type("M", (), {"content": mock_content})()})()
    mock_response = type("Resp", (), {"choices": [mock_choice]})()
    mock_client = type("Client", (), {
        "chat": type("Chat", (), {
            "completions": type("Comp", (), {
                "create": staticmethod(lambda **_: mock_response)
            })()
        })()
    })()

    import sys
    import types
    groq_mod = types.ModuleType("groq")
    groq_mod.Groq = lambda api_key: mock_client
    sys.modules["groq"] = groq_mod

    from importlib import reload
    import app.services.llm_service as llm_mod
    reload(llm_mod)

    result = llm_mod.generate_draft("concept_a", "concept_b")
    assert "AI-Generated Draft" in result


# ── ML service feature tests ──────────────────────────────────────────────────

def _make_small_graph():
    import networkx as nx
    G = nx.Graph()
    G.add_node("a", freq=10, year_freq={2020: 3, 2021: 4, 2022: 3})
    G.add_node("b", freq=5, year_freq={2020: 1, 2021: 2, 2022: 2})
    G.add_node("c", freq=8, year_freq={2020: 2, 2021: 3, 2022: 3})
    G.add_edge("a", "b", weight=3, year_weights={2020: 1, 2021: 1, 2022: 1}, articles=[1, 2])
    G.add_edge("b", "c", weight=2, year_weights={2020: 1, 2021: 1}, articles=[3])
    return G


def test_compute_graph_features_shape():
    from app.services.ml_service import _compute_graph_features

    G = _make_small_graph()
    pairs = [("a", "c"), ("a", "b")]
    X, names = _compute_graph_features(pairs, G)
    assert X.shape == (2, 16), f"Expected (2,16) got {X.shape}"
    assert len(names) == 16


def test_compute_graph_features_known_values():
    from app.services.ml_service import _compute_graph_features

    G = _make_small_graph()
    pairs = [("a", "b")]
    X, names = _compute_graph_features(pairs, G)
    # co-occurrence count for existing edge (a,b) should be the weight
    cooc_idx = names.index("cooc_count")
    assert X[0, cooc_idx] == pytest.approx(3.0)


def test_compute_embedding_features():
    from app.services.ml_service import _compute_embedding_features

    dim = 8
    embs = {
        "a": np.ones(dim, dtype=np.float32) / np.sqrt(dim),
        "b": np.zeros(dim, dtype=np.float32),
        "b": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }
    pairs = [("a", "b")]
    X, names = _compute_embedding_features(pairs, embs)
    assert X.shape == (1, 2)
    assert names == ["emb_cosine", "emb_l2"]
    # cosine between [1/√8,...] and [1,0,...] = 1/√8 ≈ 0.354
    assert 0.2 < float(X[0, 0]) < 0.5


def test_compute_sparse_features_shape():
    from app.services.ml_service import _compute_sparse_features

    G = _make_small_graph()
    node_to_idx = {n: i for i, n in enumerate(sorted(G.nodes()))}
    pairs = [("a", "c")]
    sparse_years = [2020, 2021, 2022]
    X, names = _compute_sparse_features(pairs, G, sparse_years, node_to_idx)
    assert X.shape == (1, 15)  # 5 features × 3 years
    assert len(names) == 15
