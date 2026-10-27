"""
evaluation.py
-------------
Measure retrieval quality (Precision@K) and provide a generation quality
scaffolding for manual / automated scoring.

Typical usage::

    results = run_full_eval(
        index_path="data/feedback.index",
        meta_path="data/metadata.pkl",
        eval_path="data/eval_questions.json",
    )
    print(results)
"""

import json
import logging
from typing import List, Dict, Callable, Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_eval_questions(path: str) -> List[Dict]:
    """
    Load evaluation Q&A pairs from a JSON file.

    Expected JSON format::

        [
          {
            "question": "What are common wifi complaints?",
            "relevant_ids": ["FB0012", "FB0045"],   // ground-truth feedback IDs
            "reference_answer": "Students report ..."  // optional
          },
          ...
        ]

    Parameters
    ----------
    path : str
        Path to the JSON evaluation file.

    Returns
    -------
    list of dict
        Each dict has at least ``question`` and ``relevant_ids`` keys.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid JSON or is missing required keys.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("Evaluation file must contain a JSON array.")

    for i, item in enumerate(data):
        if "question" not in item:
            raise ValueError(f"Item {i} is missing the 'question' key.")
        if "relevant_ids" not in item:
            raise ValueError(f"Item {i} is missing the 'relevant_ids' key.")

    logger.info("Loaded %d evaluation questions from %s.", len(data), path)
    return data


def evaluate_retrieval(
    questions: List[Dict],
    retrieve_fn: Callable[[str], List[Dict]],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Compute Precision@K for a set of evaluation questions.

    Precision@K = (# relevant docs in top-K) / K

    Parameters
    ----------
    questions : list of dict
        Evaluation Q&A pairs with ``question`` and ``relevant_ids`` fields.
    retrieve_fn : callable
        Function ``f(query: str) -> list[dict]``; each dict must contain
        a ``feedback_id`` key.
    k : int
        Number of results to retrieve per query.

    Returns
    -------
    dict
        ``{"precision_at_k": float, "k": int, "num_questions": int,
           "per_question": list[dict]}``
    """
    per_question = []
    precisions = []

    for item in questions:
        query = item["question"]
        relevant = set(item.get("relevant_ids", []))

        results = retrieve_fn(query)[:k]
        retrieved_ids = {r.get("feedback_id") for r in results}

        hits = len(relevant & retrieved_ids)
        precision = hits / k if k > 0 else 0.0

        per_question.append(
            {
                "question": query,
                "precision_at_k": precision,
                "hits": hits,
                "retrieved_ids": list(retrieved_ids),
                "relevant_ids": list(relevant),
            }
        )
        precisions.append(precision)

    mean_precision = sum(precisions) / len(precisions) if precisions else 0.0
    logger.info(
        "Retrieval eval: Mean Precision@%d = %.4f over %d questions.",
        k,
        mean_precision,
        len(questions),
    )
    return {
        "precision_at_k": mean_precision,
        "k": k,
        "num_questions": len(questions),
        "per_question": per_question,
    }


def evaluate_generation(
    qa_pairs: List[Dict],
    rag_fn: Callable[[str], Dict],
) -> pd.DataFrame:
    """
    Run the RAG pipeline over a set of Q&A pairs and return a scoring scaffold.

    The returned DataFrame contains the question, reference answer, generated
    answer, and empty ``manual_score`` / ``notes`` columns for human review.

    Parameters
    ----------
    qa_pairs : list of dict
        Evaluation pairs. Each must have a ``question`` key;
        ``reference_answer`` is optional.
    rag_fn : callable
        Function ``f(question: str) -> dict`` where the dict contains at least
        an ``"answer"`` key.

    Returns
    -------
    pd.DataFrame
        Columns: ``question``, ``reference_answer``, ``generated_answer``,
        ``manual_score``, ``notes``.
    """
    rows = []
    for item in qa_pairs:
        question = item["question"]
        reference = item.get("reference_answer", "")
        try:
            result = rag_fn(question)
            generated = result.get("answer", "")
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG pipeline error for question '%s': %s", question, exc)
            generated = f"ERROR: {exc}"

        rows.append(
            {
                "question": question,
                "reference_answer": reference,
                "generated_answer": generated,
                "manual_score": None,  # To be filled by human reviewer
                "notes": "",
            }
        )

    df = pd.DataFrame(rows, columns=["question", "reference_answer", "generated_answer", "manual_score", "notes"])
    logger.info("Generation eval scaffold created for %d questions.", len(df))
    return df


def run_full_eval(
    index_path: str,
    meta_path: str,
    eval_path: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run the complete evaluation suite: load resources, evaluate retrieval,
    and produce generation scaffolding.

    Parameters
    ----------
    index_path : str
        Path to the FAISS index.
    meta_path : str
        Path to the metadata pickle.
    eval_path : str
        Path to the JSON evaluation questions file.
    top_k : int
        Number of results for Precision@K.

    Returns
    -------
    dict
        ``{"retrieval": dict, "generation_scaffold": pd.DataFrame}``
    """
    from src.retrieval import load_resources, retrieve, format_context  # noqa: F401
    from src.generation import get_llm_client, rag_pipeline

    index, metadata, model = load_resources(index_path, meta_path)

    def retriever(query: str) -> List[Dict]:
        return retrieve(query, top_k=top_k, index=index, metadata=metadata, model=model)

    questions = load_eval_questions(eval_path)

    # Retrieval evaluation
    retrieval_results = evaluate_retrieval(questions, retriever, k=top_k)

    # Generation scaffold (only if API key available)
    try:
        client = get_llm_client()

        def rag_fn(question: str) -> Dict:
            return rag_pipeline(question, retriever, client)

        gen_scaffold = evaluate_generation(questions, rag_fn)
    except EnvironmentError as exc:
        logger.warning("Skipping generation eval: %s", exc)
        gen_scaffold = pd.DataFrame()

    return {
        "retrieval": retrieval_results,
        "generation_scaffold": gen_scaffold,
    }
