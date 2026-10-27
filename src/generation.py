"""
generation.py
-------------
RAG pipeline using Google Gemini API for answer generation, summaries,
and recommendations.

Requires a valid ``GEMINI_API_KEY`` in the environment (or in a ``.env`` file).
"""

import os
import logging
from typing import List, Callable, Dict, Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-1.5-flash"

_RAG_SYSTEM_INSTRUCTION = (
    "You are CampusMind, an AI assistant that helps college administrators "
    "understand student feedback. Answer questions ONLY using the context "
    "provided below. If the context does not contain enough information to "
    "answer the question, respond with: "
    "'I do not have enough data in the current feedback to answer this question.'"
)


def get_llm_client(model_name: str = None):
    """
    Create and return a configured Google Gemini GenerativeModel client.
    Automatically discovers and uses an active model that supports generateContent.
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or Streamlit Cloud Secrets."
        )

    genai.configure(api_key=api_key)

    chosen_model = model_name
    if not chosen_model:
        try:
            supported = [
                m.name for m in genai.list_models()
                if "generateContent" in getattr(m, "supported_generation_methods", [])
            ]
            flash_models = [m for m in supported if "flash" in m]
            if flash_models:
                chosen_model = flash_models[0]
            elif supported:
                chosen_model = supported[0]
        except Exception as exc:
            logger.warning("Dynamic model discovery failed: %s", exc)

    if not chosen_model:
        chosen_model = "gemini-1.5-flash-latest"

    try:
        model = genai.GenerativeModel(
            model_name=chosen_model,
            system_instruction=_RAG_SYSTEM_INSTRUCTION,
        )
    except Exception:
        model = genai.GenerativeModel(model_name=chosen_model)

    logger.info("Gemini client initialised with model: %s", chosen_model)
    return model


def _generate_with_fallback(prompt: str, client, system_instruction: str = _RAG_SYSTEM_INSTRUCTION) -> str:
    """Execute content generation with automatic model fallback if 404 occurs."""
    try:
        response = client.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        err_str = str(exc).lower()
        if "404" in err_str or "not found" in err_str or "not supported" in err_str:
            import google.generativeai as genai
            discovered = []
            try:
                for m in genai.list_models():
                    if "generateContent" in getattr(m, "supported_generation_methods", []):
                        discovered.append(m.name)
            except Exception:
                pass

            candidates = discovered + [
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro",
                "gemini-1.5-pro-latest",
                "gemini-2.0-flash-exp",
                "gemini-pro",
            ]
            for cand in candidates:
                try:
                    try:
                        fallback_model = genai.GenerativeModel(
                            model_name=cand,
                            system_instruction=system_instruction,
                        )
                    except Exception:
                        fallback_model = genai.GenerativeModel(model_name=cand)
                    res = fallback_model.generate_content(prompt)
                    return res.text.strip()
                except Exception:
                    continue
        raise exc


def build_rag_prompt(question: str, context: str) -> str:
    """
    Build a RAG prompt combining the retrieved context and user question.
    """
    prompt = (
        f"## Student Feedback Context\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"## Question\n\n"
        f"{question}\n\n"
        f"## Instructions\n\n"
        f"Answer the question using ONLY the feedback provided above. "
        f"Reference specific feedback IDs where relevant. "
        f"If the context is insufficient, say so explicitly."
    )
    return prompt


def generate_answer(question: str, context: str, client) -> str:
    """
    Generate an answer to a question using the RAG prompt.
    """
    prompt = build_rag_prompt(question, context)
    try:
        return _generate_with_fallback(prompt, client)
    except Exception as exc:
        logger.error("Gemini generate_answer failed: %s", exc)
        return f"Error generating answer: {exc}"


def generate_summary(category: str, feedback_items: List[str], client) -> str:
    """
    Generate a summary of feedback items for a given category.
    """
    if not feedback_items:
        return f"No feedback available for category: {category}."

    bullets = "\n".join(f"- {item}" for item in feedback_items[:30])
    prompt = (
        f"The following are student feedback items for the '{category}' category "
        f"at a college campus:\n\n"
        f"{bullets}\n\n"
        f"Write a concise 2-4 paragraph summary highlighting the main themes, "
        f"common complaints, and any positive observations. "
        f"Base your summary ONLY on the feedback above."
    )
    try:
        return _generate_with_fallback(prompt, client)
    except Exception as exc:
        logger.error("Gemini generate_summary failed: %s", exc)
        return f"Error generating summary: {exc}"


def generate_recommendations(summary: str, client) -> str:
    """
    Generate actionable recommendations based on a feedback summary.
    """
    prompt = (
        f"Based on the following summary of student feedback:\n\n"
        f"{summary}\n\n"
        f"Provide 3-5 specific, actionable recommendations that college administrators "
        f"can implement to address the issues raised. Format as a numbered list. "
        f"Be concrete and realistic."
    )
    try:
        return _generate_with_fallback(prompt, client)
    except Exception as exc:
        logger.error("Gemini generate_recommendations failed: %s", exc)
        return f"Error generating recommendations: {exc}"


def rag_pipeline(
    question: str,
    retriever_fn: Callable[[str], list],
    client,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve relevant feedback, build prompt, generate answer.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    retriever_fn : callable
        A function that accepts a query string and returns a list of
        feedback record dicts (as returned by :func:`src.retrieval.retrieve`).
    client : google.generativeai.GenerativeModel
        Initialised Gemini client.

    Returns
    -------
    dict
        Keys:
        - ``"question"`` – the original question
        - ``"retrieved_docs"`` – list of retrieved feedback records
        - ``"context"`` – formatted context string
        - ``"answer"`` – generated answer string
    """
    from src.retrieval import format_context

    retrieved = retriever_fn(question)
    context = format_context(retrieved)
    answer = generate_answer(question, context, client)

    return {
        "question": question,
        "retrieved_docs": retrieved,
        "context": context,
        "answer": answer,
    }
