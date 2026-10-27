"""
retrieval.py
------------
Semantic search over the FAISS feedback index.

Typical usage::

    index, metadata, model = load_resources(
        index_path="data/feedback.index",
        meta_path="data/metadata.pkl",
    )
    results = retrieve("poor wifi in hostel", top_k=5,
                       index=index, metadata=metadata, model=model)
    context = format_context(results)
"""

import pickle
import logging
from typing import List, Dict, Tuple, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_resources(
    index_path: str,
    meta_path: str,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> Tuple[Any, List[Dict], Any]:
    """
    Load the FAISS index, metadata, and embedding model from disk.

    Parameters
    ----------
    index_path : str
        Path to the saved FAISS ``.index`` file.
    meta_path : str
        Path to the saved metadata pickle file.
    model_name : str
        Name of the SentenceTransformer model to load.

    Returns
    -------
    tuple
        ``(faiss_index, metadata_list, sentence_transformer_model)``
    """
    from src.embeddings import load_model, load_index

    index = load_index(index_path)
    with open(meta_path, "rb") as fh:
        metadata: List[Dict] = pickle.load(fh)
    model = load_model(model_name)
    logger.info(
        "Resources loaded: %d vectors in index, %d metadata records.",
        index.ntotal,
        len(metadata),
    )
    return index, metadata, model


def retrieve(
    query: str,
    top_k: int = 5,
    index: Optional[Any] = None,
    metadata: Optional[List[Dict]] = None,
    model: Optional[Any] = None,
    index_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> List[Dict]:
    """
    Run a semantic nearest-neighbour search and return the top-k feedback records.

    You can either pass pre-loaded *index*, *metadata*, and *model*, **or**
    provide *index_path* / *meta_path* / *model_name* for lazy loading.

    Parameters
    ----------
    query : str
        Natural-language search query.
    top_k : int
        Number of results to return.
    index : faiss index, optional
        Pre-loaded FAISS index.
    metadata : list of dict, optional
        Pre-loaded metadata records aligned with the index.
    model : SentenceTransformer, optional
        Pre-loaded embedding model.
    index_path : str, optional
        Path to FAISS index (used when *index* is not provided).
    meta_path : str, optional
        Path to metadata pickle (used when *metadata* is not provided).
    model_name : str
        Model name for lazy loading.

    Returns
    -------
    list of dict
        Each dict is a metadata record augmented with a ``score`` key
        (L2 distance; lower = more similar).
    """
    # Lazy loading if resources not provided
    if index is None or metadata is None or model is None:
        if not index_path or not meta_path:
            raise ValueError(
                "Provide either (index, metadata, model) or (index_path, meta_path)."
            )
        index, metadata, model = load_resources(index_path, meta_path, model_name)

    # Embed query
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # Search
    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_vector, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue  # FAISS sentinel for missing results
        record = dict(metadata[idx])
        record["score"] = float(dist)
        results.append(record)

    logger.debug("Query '%s' returned %d results.", query, len(results))
    return results


def format_context(results: List[Dict]) -> str:
    """
    Format a list of retrieval results into a numbered context string
    suitable for inclusion in an LLM prompt.

    Parameters
    ----------
    results : list of dict
        Retrieved feedback records (each must contain at minimum
        ``feedback_id``, ``category``, ``department``, ``severity``,
        and ``feedback_text``).

    Returns
    -------
    str
        Multi-line context string with one record per block.
    """
    if not results:
        return "No relevant feedback found."

    lines = []
    for i, r in enumerate(results, start=1):
        fid = r.get("feedback_id", "N/A")
        cat = r.get("category", "N/A")
        dept = r.get("department", "N/A")
        sev = r.get("severity", "N/A")
        text = r.get("feedback_text", "")
        lines.append(
            f"[{i}] ID={fid} | Category={cat} | Dept={dept} | Severity={sev}\n"
            f"    Feedback: {text}"
        )
    return "\n\n".join(lines)
