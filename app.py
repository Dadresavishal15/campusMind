"""
app.py
------
CampusMind – 3-page Streamlit dashboard.

Pages:
  1. Overview    – Stats, category & severity charts, top themes
  2. Ask & Search – RAG-powered Q&A with source expansion
  3. Reports      – Per-category summary + recommendations via Gemini

Run:
    streamlit run app.py
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Environment & Paths ───────────────────────────────────────────────────────
load_dotenv()
ROOT = Path(__file__).parent
DATA_CSV = ROOT / "data" / "synthetic_feedback.csv"
INDEX_PATH = str(ROOT / "data" / "feedback.index")
META_PATH = str(ROOT / "data" / "metadata.pkl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CampusMind",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_feedback() -> pd.DataFrame:
    """Load and cache the synthetic feedback CSV; generate if missing."""
    if not DATA_CSV.exists():
        try:
            from data.generate_synthetic import generate_dataset
            DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
            df = generate_dataset(400)
            df.to_csv(DATA_CSV, index=False)
            return df
        except Exception as exc:
            st.error(f"Failed to auto-generate dataset: {exc}")
            st.stop()
    return pd.read_csv(DATA_CSV)


@st.cache_resource
def get_embedding_model():
    """Load and cache the sentence-transformer model."""
    try:
        from src.embeddings import load_model
        return load_model()
    except ImportError as exc:
        st.error(f"sentence-transformers not installed: {exc}")
        return None


@st.cache_resource
def get_faiss_resources():
    """Load FAISS index and metadata; build if missing."""
    import pickle

    if not Path(INDEX_PATH).exists() or not Path(META_PATH).exists():
        st.info("Building FAISS index for the first time … this may take a minute.")
        try:
            from src.embeddings import build_and_save
            df = load_feedback()
            build_and_save(df, INDEX_PATH, META_PATH)
        except Exception as exc:
            st.error(f"Failed to build index: {exc}")
            return None, None

    try:
        from src.embeddings import load_index
        import pickle

        index = load_index(INDEX_PATH)
        with open(META_PATH, "rb") as fh:
            metadata = pickle.load(fh)
        return index, metadata
    except Exception as exc:
        st.error(f"Failed to load index: {exc}")
        return None, None


def get_gemini_client():
    """Return Gemini client or None with a warning, checking st.secrets and env."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
                os.environ["GEMINI_API_KEY"] = api_key
        except Exception:
            pass

    if not api_key:
        st.warning(
            "⚠️ **GEMINI_API_KEY not set.** "
            "Add your key to `.env` or Streamlit Cloud Secrets to enable AI-generated answers. "
            "Retrieval still works without a key."
        )
        return None
    try:
        from src.generation import get_llm_client
        return get_llm_client()
    except Exception as exc:
        st.error(f"Failed to initialise Gemini client: {exc}")
        return None


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
    st.title("CampusMind 🎓")
    st.markdown("*Student Experience Intelligence*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Overview", "💬 Ask & Search", "📄 Reports"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Powered by Gemini + FAISS + SentenceTransformers")

df = load_feedback()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Campus Feedback Overview")
    st.markdown(
        "A bird's-eye view of student feedback across all departments and categories."
    )

    # ── KPI row ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responses", f"{len(df):,}")
    col2.metric("Departments", df["department"].nunique())
    col3.metric("Categories", df["category"].nunique())
    high_pct = round((df["severity"] == "high").mean() * 100, 1)
    col4.metric("High Severity %", f"{high_pct}%")

    st.divider()

    # ── Charts row ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Feedback by Category")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        st.bar_chart(cat_counts.set_index("Category"))

    with chart_col2:
        st.subheader("Severity Distribution")
        sev_counts = df["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        # Use a simple table since st.pie_chart is not built-in
        st.bar_chart(sev_counts.set_index("Severity"))

    st.divider()

    # ── Department breakdown ──
    st.subheader("Feedback by Department")
    dept_cat = (
        df.groupby(["department", "category"])
        .size()
        .reset_index(name="count")
        .pivot(index="department", columns="category", values="count")
        .fillna(0)
        .astype(int)
    )
    st.dataframe(dept_cat, use_container_width=True)

    st.divider()

    # ── Recent feedback ──
    st.subheader("Recent Feedback Samples")
    severity_filter = st.selectbox(
        "Filter by severity", ["all", "high", "medium", "low"], index=0
    )
    sample_df = df if severity_filter == "all" else df[df["severity"] == severity_filter]
    st.dataframe(
        sample_df[["feedback_id", "date", "department", "category", "severity", "feedback_text"]]
        .sort_values("date", ascending=False)
        .head(20),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – ASK & SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Ask & Search":
    st.title("💬 Ask & Search")
    st.markdown(
        "Ask a question about campus feedback. The system retrieves the most "
        "relevant feedback items and uses Gemini to generate an answer."
    )

    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the main complaints about the hostel?",
    )

    top_k = st.slider("Number of source documents", min_value=3, max_value=10, value=5)

    if st.button("🔍 Search & Answer", disabled=not question.strip()):
        index, metadata = get_faiss_resources()
        model = get_embedding_model()

        if index is None or model is None:
            st.error("Index or model not available. Check setup.")
        else:
            with st.spinner("Retrieving relevant feedback …"):
                try:
                    from src.retrieval import retrieve, format_context

                    results = retrieve(
                        question,
                        top_k=top_k,
                        index=index,
                        metadata=metadata,
                        model=model,
                    )
                    context = format_context(results)
                except Exception as exc:
                    st.error(f"Retrieval failed: {exc}")
                    st.stop()

            # ── Gemini answer ──
            client = get_gemini_client()
            if client:
                with st.spinner("Generating answer with Gemini …"):
                    try:
                        from src.generation import generate_answer

                        answer = generate_answer(question, context, client)
                        st.success("**Answer**")
                        st.markdown(answer)
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")
            else:
                st.info("ℹ️ No Gemini API key — showing retrieved documents only.")

            # ── Source snippets ──
            st.divider()
            st.subheader(f"📎 Source Documents ({len(results)} retrieved)")
            for i, rec in enumerate(results, start=1):
                with st.expander(
                    f"[{i}] {rec.get('feedback_id')} | {rec.get('category')} | "
                    f"{rec.get('department')} | Severity: {rec.get('severity')} "
                    f"| Score: {rec.get('score', 0):.4f}"
                ):
                    st.markdown(f"> {rec.get('feedback_text', '')}")
                    st.caption(
                        f"Semester {rec.get('semester')} · {rec.get('date')}"
                    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Reports":
    st.title("📄 Category Reports")
    st.markdown(
        "Select a feedback category to generate an AI-powered summary and "
        "recommended actions."
    )

    categories = sorted(df["category"].unique().tolist())
    selected_cat = st.selectbox("Select category", categories)

    if st.button("🤖 Generate Report"):
        category_df = df[df["category"] == selected_cat]
        feedback_items = category_df["feedback_text"].tolist()

        st.subheader(f"Category: **{selected_cat.title()}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Responses", len(category_df))
        col2.metric(
            "High Severity",
            (category_df["severity"] == "high").sum(),
        )
        col3.metric("Departments Affected", category_df["department"].nunique())

        st.divider()

        client = get_gemini_client()
        if client:
            with st.spinner("Generating summary …"):
                try:
                    from src.generation import generate_summary, generate_recommendations

                    summary = generate_summary(selected_cat, feedback_items, client)
                    st.subheader("📝 Summary")
                    st.markdown(summary)

                    st.divider()

                    with st.spinner("Generating recommendations …"):
                        recommendations = generate_recommendations(summary, client)
                        st.subheader("✅ Recommended Actions")
                        st.markdown(recommendations)
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")
        else:
            st.info(
                "ℹ️ Set `GEMINI_API_KEY` in your `.env` file to generate "
                "AI-powered summaries and recommendations."
            )
            st.subheader("📋 Raw Feedback Samples")
            st.dataframe(
                category_df[["feedback_id", "department", "severity", "feedback_text"]].head(20),
                use_container_width=True,
            )
