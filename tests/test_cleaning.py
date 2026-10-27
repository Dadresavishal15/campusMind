"""
test_cleaning.py
----------------
Unit tests for src/cleaning.py
"""

import sys
import os
import io
import pytest
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cleaning import anonymize_text, normalize_categories, clean_pipeline, load_raw


# ── anonymize_text ─────────────────────────────────────────────────────────────

def test_anonymize_removes_email():
    """Email addresses must be replaced with [EMAIL]."""
    text = "Please contact student.helpdesk@campus.edu for more info."
    result = anonymize_text(text)
    assert "[EMAIL]" in result
    assert "@" not in result


def test_anonymize_removes_email_multiple():
    """Multiple email addresses in a single string must all be redacted."""
    text = "Forward to admin@uni.ac.in and hod@dept.edu please."
    result = anonymize_text(text)
    assert result.count("[EMAIL]") == 2
    assert "@" not in result


def test_anonymize_removes_phone_number_10_digit():
    """10-digit Indian mobile numbers must be replaced with [PHONE]."""
    text = "Call us at 9876543210 for assistance."
    result = anonymize_text(text)
    assert "[PHONE]" in result
    assert "9876543210" not in result


def test_anonymize_removes_student_roll_number():
    """Student roll numbers matching the pattern must be replaced with [ID]."""
    text = "Student 21CSE0042 submitted the complaint."
    result = anonymize_text(text)
    assert "[ID]" in result
    assert "21CSE0042" not in result


def test_anonymize_preserves_normal_text():
    """Normal text without PII must pass through unchanged."""
    text = "The library opening hours are too short."
    result = anonymize_text(text)
    assert result == text


def test_anonymize_handles_non_string():
    """Non-string input (e.g. None) must be returned as-is without raising."""
    assert anonymize_text(None) is None
    assert anonymize_text(42) == 42


# ── normalize_categories ───────────────────────────────────────────────────────

def test_normalize_categories_lowercases():
    """Categories must be lowercased."""
    df = pd.DataFrame({"category": ["Academics", "HOSTEL", "Canteen"]})
    result = normalize_categories(df)
    assert list(result["category"]) == ["academics", "hostel", "canteen"]


def test_normalize_categories_handles_unknown():
    """Unknown category values must be mapped to 'other'."""
    df = pd.DataFrame({"category": ["academics", "swimming_pool", "unknown_cat"]})
    result = normalize_categories(df)
    unknown_rows = result[result["category"] == "other"]
    assert len(unknown_rows) == 2


def test_normalize_categories_does_not_mutate_original():
    """The original DataFrame must not be modified (copy semantics)."""
    df = pd.DataFrame({"category": ["Academics"]})
    original_value = df.at[0, "category"]
    _ = normalize_categories(df)
    assert df.at[0, "category"] == original_value


# ── clean_pipeline ─────────────────────────────────────────────────────────────

def _make_temp_csv(tmp_path, rows=None):
    """Helper to write a temporary CSV for testing."""
    if rows is None:
        rows = [
            {
                "feedback_id": "FB0001",
                "department": "CSE",
                "semester": 3,
                "category": "academics",
                "feedback_text": "The lab sessions are helpful.",
                "severity": "low",
                "date": "2024-09-01",
            },
            {
                "feedback_id": "FB0002",
                "department": "ME",
                "semester": 5,
                "category": "canteen",
                "feedback_text": "Contact canteen@campus.edu for menu.",
                "severity": "medium",
                "date": "2024-10-15",
            },
        ]
    path = tmp_path / "test_feedback.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_clean_pipeline_returns_dataframe(tmp_path):
    """clean_pipeline must return a pd.DataFrame."""
    csv_path = _make_temp_csv(tmp_path)
    result = clean_pipeline(csv_path)
    assert isinstance(result, pd.DataFrame)


def test_clean_pipeline_expected_columns(tmp_path):
    """Output DataFrame must contain all expected columns."""
    csv_path = _make_temp_csv(tmp_path)
    result = clean_pipeline(csv_path)
    expected_cols = {"feedback_id", "department", "semester", "category", "feedback_text", "severity", "date"}
    assert expected_cols.issubset(set(result.columns))


def test_clean_pipeline_anonymizes_email(tmp_path):
    """Emails in feedback_text must be stripped by clean_pipeline."""
    csv_path = _make_temp_csv(tmp_path)
    result = clean_pipeline(csv_path)
    for text in result["feedback_text"]:
        assert "@" not in text


def test_clean_pipeline_drops_invalid_semesters(tmp_path):
    """Rows with out-of-range semesters must be dropped."""
    rows = [
        {
            "feedback_id": "FB0001", "department": "CSE", "semester": 9,
            "category": "academics", "feedback_text": "Test feedback.",
            "severity": "low", "date": "2024-09-01",
        },
        {
            "feedback_id": "FB0002", "department": "ECE", "semester": 3,
            "category": "canteen", "feedback_text": "Good food today.",
            "severity": "medium", "date": "2024-10-01",
        },
    ]
    csv_path = _make_temp_csv(tmp_path, rows=rows)
    result = clean_pipeline(csv_path)
    assert len(result) == 1
    assert result.iloc[0]["feedback_id"] == "FB0002"
