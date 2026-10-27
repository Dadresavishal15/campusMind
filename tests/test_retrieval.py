"""
test_retrieval.py
-----------------
Unit tests for src/retrieval.py
"""

import sys
import os
import pickle
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval import retrieve, format_context


# ── Fixtures ──────────────────────────────────────────────────────────────────

class _FakeModel:
    """Minimal stub that mimics SentenceTransformer.encode()."""

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        # Return deterministic random-ish embeddings based on text length
        rng = np.random.default_rng(seed=sum(len(t) for t in texts))
        return rng.random((len(texts), 4)).astype(np.float32)


def _build_fake_index(records):
    """Build a real FAISS index populated with fake embeddings."""
    import faiss

    model = _FakeModel()
    texts = [r["feedback_text"] for r in records]
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(4)
    index.add(embeddings)
    return index


_SAMPLE_RECORDS = [
    {
        "feedback_id": "FB0001",
        "department": "CSE",
        "semester": 3,
        "category": "wifi",
        "feedback_text": "Wi-Fi in the hostel is very slow.",
        "severity": "high",
        "date": "2024-09-01",
    },
    {
        "feedback_id": "FB0002",
        "department": "ECE",
        "semester": 5,
        "category": "canteen",
        "feedback_text": "The canteen food quality has declined.",
        "severity": "medium",
        "date": "2024-10-01",
    },
    {
        "feedback_id": "FB0003",
        "department": "ME",
        "semester": 2,
        "category": "library",
        "feedback_text": "Library books are outdated.",
        "severity": "low",
        "date": "2025-01-15",
    },
    {
        "feedback_id": "FB0004",
        "department": "CE",
        "semester": 4,
        "category": "academics",
        "feedback_text": "Lectures are well-structured.",
        "severity": "low",
        "date": "2025-02-20",
    },
    {
        "feedback_id": "FB0005",
        "department": "MBA",
        "semester": 1,
        "category": "hostel",
        "feedback_text": "Hot water supply is unreliable.",
        "severity": "high",
        "date": "2025-03-10",
    },
]


@pytest.fixture
def fake_resources():
    """Return (index, metadata, model) built from _SAMPLE_RECORDS."""
    pytest.importorskip("faiss", reason="faiss-cpu required for retrieval tests")
    index = _build_fake_index(_SAMPLE_RECORDS)
    return index, list(_SAMPLE_RECORDS), _FakeModel()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_retrieve_returns_results(fake_resources):
    """retrieve() must return a non-empty list for a valid query."""
    index, metadata, model = fake_resources
    results = retrieve(
        "slow internet wifi",
        top_k=3,
        index=index,
        metadata=metadata,
        model=model,
    )
    assert isinstance(results, list)
    assert len(results) > 0


def test_retrieve_returns_correct_number_of_results(fake_resources):
    """retrieve() must return at most top_k results."""
    index, metadata, model = fake_resources
    for k in [1, 3, 5]:
        results = retrieve(
            "student feedback",
            top_k=k,
            index=index,
            metadata=metadata,
            model=model,
        )
        assert len(results) <= k


def test_retrieve_results_contain_score(fake_resources):
    """Each result dict must contain a 'score' key."""
    index, metadata, model = fake_resources
    results = retrieve(
        "canteen food quality",
        top_k=2,
        index=index,
        metadata=metadata,
        model=model,
    )
    for r in results:
        assert "score" in r
        assert isinstance(r["score"], float)


def test_retrieve_results_contain_feedback_id(fake_resources):
    """Each result dict must contain a 'feedback_id' key."""
    index, metadata, model = fake_resources
    results = retrieve("library books", top_k=3, index=index, metadata=metadata, model=model)
    for r in results:
        assert "feedback_id" in r


def test_retrieve_raises_without_resources():
    """retrieve() must raise ValueError when neither pre-loaded nor path resources given."""
    with pytest.raises(ValueError):
        retrieve("test query")


# ── format_context ─────────────────────────────────────────────────────────────

def test_format_context_includes_feedback_ids(fake_resources):
    """format_context() output must include the feedback_id of each result."""
    index, metadata, model = fake_resources
    results = retrieve("hostel hot water", top_k=3, index=index, metadata=metadata, model=model)
    context = format_context(results)
    for r in results:
        fid = r.get("feedback_id", "")
        assert fid in context, f"Expected feedback_id {fid} in context."


def test_format_context_empty_returns_no_feedback_message():
    """format_context() with an empty list must return a 'no feedback' message."""
    context = format_context([])
    assert "no relevant feedback" in context.lower() or "not found" in context.lower() or "No relevant" in context


def test_format_context_structure(fake_resources):
    """format_context() must include category and department in the output."""
    index, metadata, model = fake_resources
    results = retrieve("wifi speed issue", top_k=2, index=index, metadata=metadata, model=model)
    context = format_context(results)
    # Should contain at least one category token from the results
    assert any(r["category"] in context for r in results)
