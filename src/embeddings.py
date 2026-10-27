"""
embeddings.py
-------------
Create sentence embeddings and build a FAISS vector index over feedback data.

Uses the ``all-MiniLM-L6-v2`` model (downloaded on first run, then cached).
No API key required — inference runs entirely on-device.
"""

import os
import pickle
import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_model(model_name: str = MODEL_NAME):
    """
    Load and return a SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Defaults to ``'all-MiniLM-L6-v2'``.

    Returns
    -------
    SentenceTransformer
        Loaded embedding model.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc

    logger.info("Loading SentenceTransformer model: %s", model_name)
    model = SentenceTransformer(model_name)
    return model


def embed_texts(texts: List[str], model) -> np.ndarray:
    """
    Compute sentence embeddings for a list of text strings.

    Parameters
    ----------
    texts : list of str
        Texts to embed.
    model : SentenceTransformer
        Pre-loaded embedding model.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(len(texts), embedding_dim)``.
    """
    logger.info("Embedding %d texts …", len(texts))
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS ``IndexFlatL2`` from a matrix of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Float32 matrix of shape ``(n, d)`` where *d* is the embedding dimension.

    Returns
    -------
    faiss.IndexFlatL2
        FAISS index with all vectors added.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is not installed. Run: pip install faiss-cpu"
        ) from exc

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logger.info("FAISS index built with %d vectors (dim=%d).", index.ntotal, dimension)
    return index


def save_index(index, path: str) -> None:
    """
    Persist a FAISS index to disk.

    Parameters
    ----------
    index : faiss.IndexFlatL2
        FAISS index to save.
    path : str
        Destination file path (e.g. ``'data/feedback.index'``).
    """
    import faiss

    faiss.write_index(index, path)
    logger.info("FAISS index saved to %s", path)


def load_index(path: str):
    """
    Load a FAISS index from disk.

    Parameters
    ----------
    path : str
        Path to the saved FAISS index file.

    Returns
    -------
    faiss.IndexFlatL2
        Loaded FAISS index.
    """
    import faiss

    index = faiss.read_index(path)
    logger.info("FAISS index loaded from %s (%d vectors).", path, index.ntotal)
    return index


def build_and_save(df: pd.DataFrame, index_path: str, meta_path: str) -> None:
    """
    Full pipeline: embed feedback texts, build FAISS index, and save both index
    and metadata to disk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a ``feedback_text`` column.
    index_path : str
        File path for the FAISS index (e.g. ``'data/feedback.index'``).
    meta_path : str
        File path for the metadata pickle (e.g. ``'data/metadata.pkl'``).
    """
    model = load_model()
    texts = df["feedback_text"].tolist()
    embeddings = embed_texts(texts, model)

    index = build_faiss_index(embeddings)

    # Persist index
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    save_index(index, index_path)

    # Persist metadata (all columns as list of dicts)
    metadata = df.to_dict(orient="records")
    with open(meta_path, "wb") as fh:
        pickle.dump(metadata, fh)
    logger.info("Metadata saved to %s (%d records).", meta_path, len(metadata))
