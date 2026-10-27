"""
cleaning.py
-----------
Load raw feedback CSV, anonymize PII, normalize categories, validate schema.

Pipeline: load_raw → anonymize_text → normalize_categories → validate → save_clean
"""

import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical category list
VALID_CATEGORIES = {
    "academics", "hostel", "canteen", "sports",
    "library", "wifi", "administration", "mental_health",
}

# Canonical department list
VALID_DEPARTMENTS = {"CSE", "ECE", "ME", "CE", "MBA", "BBA", "Arts", "Science"}

# Regex patterns for PII
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(\+91[\-\s]?)?[6-9]\d{9}"          # Indian mobile
    r"|(\+?1[\-\s]?)?\(?\d{3}\)?[\-\s]\d{3}[\-\s]\d{4}"  # US style
    r"|\b\d{10,12}\b",                    # Generic long numeric
    re.IGNORECASE,
)
# Student / roll number patterns (e.g. 21CSE001, 21CSE0042, 2021BBA0042)
# Matches: 2-digit year (21) or 4-digit year (2021) followed by dept code + numeric suffix
_ROLL_RE = re.compile(
    r"\b(?:20\d{2}|\d{2})[A-Z]{2,5}\d{3,5}\b",
    re.IGNORECASE,
)


def load_raw(path: str) -> pd.DataFrame:
    """
    Load a raw feedback CSV from disk.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame, with leading/trailing whitespace stripped from string columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *path*.
    ValueError
        If the CSV is missing required columns.
    """
    required_cols = {"feedback_id", "department", "semester", "category", "feedback_text", "severity", "date"}
    df = pd.read_csv(path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def anonymize_text(text: str) -> str:
    """
    Strip PII from a single feedback string.

    Removes:
    - Email addresses → ``[EMAIL]``
    - Phone numbers → ``[PHONE]``
    - Student roll numbers → ``[ID]``

    Parameters
    ----------
    text : str
        Raw feedback text.

    Returns
    -------
    str
        Anonymized feedback text.
    """
    if not isinstance(text, str):
        return text
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _ROLL_RE.sub("[ID]", text)
    return text.strip()


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the ``category`` column to lowercase and map unknowns to ``'other'``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a ``category`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the ``category`` column normalised in-place.
    """
    df = df.copy()
    df["category"] = df["category"].str.lower().str.strip()
    unknown_mask = ~df["category"].isin(VALID_CATEGORIES)
    if unknown_mask.any():
        logger.warning(
            "Found %d rows with unknown categories — mapping to 'other'.",
            unknown_mask.sum(),
        )
    df.loc[unknown_mask, "category"] = "other"
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data types and drop rows that are clearly malformed.

    Rules:
    - ``semester`` must be an integer in [1, 8].
    - ``feedback_text`` must be non-empty.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    pd.DataFrame
        Validated (possibly smaller) DataFrame.
    """
    before = len(df)
    df = df.copy()

    # Coerce semester to numeric, drop non-numeric
    df["semester"] = pd.to_numeric(df["semester"], errors="coerce")
    df = df.dropna(subset=["semester"])
    df["semester"] = df["semester"].astype(int)
    df = df[df["semester"].between(1, 8)]

    # Drop rows with empty feedback
    df = df[df["feedback_text"].str.strip().astype(bool)]

    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d malformed rows during validation.", dropped)
    return df.reset_index(drop=True)


def clean_pipeline(path: str) -> pd.DataFrame:
    """
    Full cleaning pipeline: load → anonymize → normalize → validate.

    Parameters
    ----------
    path : str
        Path to the raw CSV.

    Returns
    -------
    pd.DataFrame
        Clean, anonymized, validated DataFrame.
    """
    df = load_raw(path)
    df["feedback_text"] = df["feedback_text"].apply(anonymize_text)
    df = normalize_categories(df)
    df = _validate(df)
    logger.info("Clean pipeline complete. %d rows retained.", len(df))
    return df


def save_clean(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a cleaned DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame to persist.
    output_path : str
        Destination CSV path.
    """
    df.to_csv(output_path, index=False)
    logger.info("Clean data saved to %s", output_path)
