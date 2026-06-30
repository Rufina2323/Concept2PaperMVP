"""Tests for REST API endpoints."""

import uuid
from unittest.mock import MagicMock, patch

from app.models import ConceptPair
from app.models import Session as SessionModel


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_get_concepts(client):
    mock_concepts = ["deep network", "reinforcement learning", "transformer"]
    with patch("app.routers.concepts.get_top_concepts", return_value=mock_concepts):
        r = client.get("/api/v1/concepts")
    assert r.status_code == 200
    data = r.json()
    assert "concepts" in data
    assert data["concepts"] == mock_concepts


def test_create_session_and_get(client, db_session):
    keyword = "deep network"
    with patch("app.routers.sessions.run_inference") as mock_task:
        mock_task.delay = MagicMock()
        r = client.post("/api/v1/sessions", json={"keyword": keyword})

    assert r.status_code == 202
    data = r.json()
    assert data["keyword"] == keyword
    assert data["status"] == "pending"
    session_id = data["id"]

    r2 = client.get(f"/api/v1/sessions/{session_id}")
    assert r2.status_code == 200
    assert r2.json()["id"] == session_id


def test_get_session_not_found(client):
    r = client.get(f"/api/v1/sessions/{uuid.uuid4()}")
    assert r.status_code == 404


def test_session_with_pairs(client, db_session):
    """Session that already has pairs stored returns them."""
    session = SessionModel(keyword="transformer", status="done")
    db_session.add(session)
    db_session.flush()

    pair = ConceptPair(
        session_id=session.id,
        concept_a="attention mechanism",
        concept_b="transformer",
        score=0.95,
        rank=1,
    )
    db_session.add(pair)
    db_session.commit()

    r = client.get(f"/api/v1/sessions/{session.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert len(body["pairs"]) == 1
    assert body["pairs"][0]["score"] == 0.95


def test_create_draft_and_get(client, db_session):
    with patch("app.routers.drafts.run_draft") as mock_task:
        mock_task.delay = MagicMock()
        r = client.post(
            "/api/v1/drafts",
            json={"concept_a": "deep network", "concept_b": "transformer"},
        )
    assert r.status_code == 202
    data = r.json()
    assert data["status"] == "pending"
    draft_id = data["id"]

    r2 = client.get(f"/api/v1/drafts/{draft_id}")
    assert r2.status_code == 200
    assert r2.json()["id"] == draft_id


def test_get_draft_not_found(client):
    r = client.get(f"/api/v1/drafts/{uuid.uuid4()}")
    assert r.status_code == 404
