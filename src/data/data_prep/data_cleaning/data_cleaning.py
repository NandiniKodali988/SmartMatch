import pandas as pd
import numpy as np
import re


# ─────────────────────────────────────────
# 1. Core text cleaner
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light cleaning suitable for LLM / SigLIP downstream use.
    - Lowercase
    - Remove URLs, HTML tags
    - Remove weird symbols (keep letters, digits, basic punctuation)
    - Collapse extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)          # URLs
    text = re.sub(r"<[^>]+>", "", text)                         # HTML tags
    text = re.sub(r"[^a-z0-9\s,.\-']", " ", text)              # keep basic chars
    text = re.sub(r"\s+", " ", text).strip()                    # collapse spaces
    return text


AI_PREFIX_PATTERN = re.compile(
    r"^(a photo of|an image of|a picture of|photo of|image of)\s+", flags=re.IGNORECASE
)

def clean_ai_description(text: str) -> str:
    """Extra step for ai_description: strip generic leading phrases."""
    cleaned = clean_text(text)
    cleaned = AI_PREFIX_PATTERN.sub("", cleaned).strip()
    return cleaned


# ─────────────────────────────────────────
# 2. Clean photos table
# ─────────────────────────────────────────

def clean_photos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["photo_description_clean"] = df["photo_description"].apply(clean_text)
    df["ai_description_clean"]    = df["ai_description"].apply(clean_ai_description)

    # Convert empty strings back to NaN for NaN-logic below
    df["photo_description_clean"] = df["photo_description_clean"].replace("", np.nan)
    df["ai_description_clean"]    = df["ai_description_clean"].replace("", np.nan)

    # NaN coverage flag
    has_photo = df["photo_description_clean"].notna()
    has_ai    = df["ai_description_clean"].notna()

    df["description_source"] = np.select(
        [has_photo & has_ai,   has_photo & ~has_ai,  ~has_photo & has_ai],
        ["both",               "photo_only",          "ai_only"],
        default="none"                                    # both NaN → mark weak
    )

    return df


# ─────────────────────────────────────────
# 3. Clean keywords table
# ─────────────────────────────────────────

CONFIDENCE_THRESHOLD = 60.0

def clean_keywords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalise keyword text
    df["keyword"] = df["keyword"].str.lower().str.strip()

    # Keep row if: user-suggested OR ai confidence above threshold
    ai_conf_cols = [c for c in df.columns if "confidence" in c]

    if ai_conf_cols:
        # Fill NaN confidences with 0 so comparison is safe
        df[ai_conf_cols] = df[ai_conf_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        max_conf = df[ai_conf_cols].max(axis=1)
        keep_mask = (df["suggested_by_user"] == True) | (max_conf >= CONFIDENCE_THRESHOLD)
    else:
        keep_mask = pd.Series(True, index=df.index)

    df = df[keep_mask].copy()

    # Attach a best_confidence column for reference
    if ai_conf_cols:
        df["best_confidence"] = max_conf[keep_mask]

    # Group into keyword list per photo
    keyword_df = (
        df.groupby("photo_id")["keyword"]
        .apply(lambda kws: sorted(set(kws)))          # deduplicate, sort
        .reset_index()
        .rename(columns={"keyword": "keyword_list"})
    )

    return keyword_df


# ─────────────────────────────────────────
# 4. Join & build master table
# ─────────────────────────────────────────

def build_master(photos_df: pd.DataFrame, keywords_df: pd.DataFrame) -> pd.DataFrame:
    photos_clean   = clean_photos(photos_df)
    keywords_clean = clean_keywords(keywords_df)

    master = photos_clean.merge(keywords_clean, on="photo_id", how="left")

    # Photos with no keywords at all
    master["keyword_list"] = master["keyword_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Select & reorder final columns
    keep_cols = [
        "photo_id",
        "photo_description_clean",
        "ai_description_clean",
        "description_source",          # 'both' | 'photo_only' | 'ai_only' | 'none'
        "keyword_list",
        "stats_views",
        "stats_downloads",
    ]
    # Only keep cols that exist in the dataframe
    keep_cols = [c for c in keep_cols if c in master.columns]
    master = master[keep_cols]

    return master


# ─────────────────────────────────────────
# 5. Quick diagnostics
# ─────────────────────────────────────────

def print_diagnostics(master: pd.DataFrame):
    print("=== Master table shape:", master.shape)
    print("\n--- description_source distribution ---")
    print(master["description_source"].value_counts())
    print("\n--- Rows with no keywords ---")
    print((master["keyword_list"].apply(len) == 0).sum())
    print("\n--- Sample rows ---")
    print(master.head(3).to_string())


# ─────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    # ── Load your data ──────────────────────────────────────────────
    # Replace these paths / adjust sep / encoding as needed
    photos_df   = pd.read_csv("photos.csv")
    keywords_df = pd.read_csv("keywords.csv")

    master = build_master(photos_df, keywords_df)
    print_diagnostics(master)

    master.to_csv("master_clean.csv", index=False)
    print("\n✓ Saved → master_clean.csv")
