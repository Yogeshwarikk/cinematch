"""
model.py — CineMatch Recommendation Engine
============================================
Run this script ONCE to preprocess the TMDB dataset and
generate the two artifacts that app.py depends on:

    artifacts/movies.pkl      – cleaned DataFrame (title + tags)
    artifacts/similarity.pkl  – cosine-similarity matrix (5000×5000)

Usage
-----
    python model.py

Dataset
-------
Download from Kaggle:
  https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
You need two files in the `data/` directory:
  - tmdb_5000_movies.csv
  - tmdb_5000_credits.csv
"""

import os
import ast
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# 1. Load Raw Data
# ─────────────────────────────────────────────
def load_data(movies_path: str, credits_path: str) -> pd.DataFrame:
    """
    Load and merge the TMDB movies and credits CSVs.

    Parameters
    ----------
    movies_path  : path to tmdb_5000_movies.csv
    credits_path : path to tmdb_5000_credits.csv

    Returns
    -------
    pd.DataFrame with merged data
    """
    print("📂 Loading datasets…")
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # The credits file uses 'movie_id' or 'id'; standardise
    if "movie_id" in credits.columns:
        credits = credits.rename(columns={"movie_id": "id"})

    df = movies.merge(credits, on="id")

    # Keep only the columns we need
    df = df[["id", "title_x", "overview", "genres", "keywords", "cast", "crew"]]
    df = df.rename(columns={"title_x": "title"})

    print(f"   → {len(df):,} movies loaded after merge")
    return df


# ─────────────────────────────────────────────
# 2. Parsing Helpers
# ─────────────────────────────────────────────
def parse_list(obj, key: str = "name", limit: int = None) -> list[str]:
    """
    Convert a JSON-string column (e.g. genres, keywords, cast) into a
    plain Python list of strings.

    Parameters
    ----------
    obj   : raw cell value (string or list)
    key   : dict key to extract (e.g. 'name')
    limit : optional max items to return

    Returns
    -------
    list[str]
    """
    try:
        items = ast.literal_eval(obj)
        result = [d[key] for d in items if key in d]
        return result[:limit] if limit else result
    except (ValueError, TypeError):
        return []


def extract_director(crew_obj) -> list[str]:
    """
    Extract the director's name from the crew column.
    Returns a list (possibly empty) so it can be concatenated uniformly.
    """
    try:
        for person in ast.literal_eval(crew_obj):
            if person.get("job") == "Director":
                return [person["name"]]
    except (ValueError, TypeError):
        pass
    return []


def collapse(tokens: list[str]) -> list[str]:
    """Remove spaces within multi-word names so they become single tokens."""
    return [t.replace(" ", "") for t in tokens]


# ─────────────────────────────────────────────
# 3. Preprocessing Pipeline
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer the 'tags' feature column.

    Steps:
    1. Drop rows with missing overview
    2. Parse JSON-string columns (genres, keywords, cast, crew)
    3. Tokenise and collapse whitespace so phrases are treated as single tokens
    4. Concatenate all features into a single `tags` string per movie
    5. Lowercase everything

    Parameters
    ----------
    df : merged TMDB DataFrame

    Returns
    -------
    Clean DataFrame with columns ['id', 'title', 'tags']
    """
    print("🔧 Preprocessing…")

    # ── 3a. Drop missing overviews ──
    df = df.dropna(subset=["overview"]).copy()
    df["overview"] = df["overview"].fillna("")

    # ── 3b. Parse structured columns ──
    df["genres"]   = df["genres"].apply(lambda x: collapse(parse_list(x)))
    df["keywords"] = df["keywords"].apply(lambda x: collapse(parse_list(x)))
    df["cast"]     = df["cast"].apply(lambda x: collapse(parse_list(x, limit=5)))
    df["crew"]     = df["crew"].apply(extract_director).apply(collapse)

    # Tokenise overview (split into word list)
    df["overview_tokens"] = df["overview"].apply(
        lambda x: x.split() if isinstance(x, str) else []
    )

    # ── 3c. Build 'tags' — combined feature vector ──
    df["tags"] = (
        df["overview_tokens"]
        + df["genres"]
        + df["keywords"]
        + df["cast"]
        + df["crew"]
    ).apply(lambda tokens: " ".join(tokens).lower())

    result = df[["id", "title", "tags"]].reset_index(drop=True)
    print(f"   → {len(result):,} movies after cleaning")
    return result


# ─────────────────────────────────────────────
# 4. Vectorisation & Similarity
# ─────────────────────────────────────────────
def build_similarity(df: pd.DataFrame):
    """
    Vectorise the 'tags' column with CountVectorizer and compute
    pairwise cosine similarity.

    CountVectorizer is used instead of TF-IDF here because:
    - Movie descriptions are short → TF-IDF adds little benefit
    - We want equal weight for genre/keyword tokens (no IDF discounting)

    Parameters
    ----------
    df : preprocessed DataFrame with 'tags' column

    Returns
    -------
    similarity : np.ndarray of shape (n, n)
    """
    print("🔢 Vectorising tags…")
    vectorizer = CountVectorizer(
        max_features=5000,          # vocabulary cap
        stop_words="english",       # remove common words
    )
    vectors = vectorizer.fit_transform(df["tags"]).toarray()
    print(f"   → Vector matrix: {vectors.shape}")

    print("📐 Computing cosine similarity…")
    similarity = cosine_similarity(vectors)
    print(f"   → Similarity matrix: {similarity.shape}")
    return similarity


# ─────────────────────────────────────────────
# 5. Save Artifacts
# ─────────────────────────────────────────────
def save_artifacts(df: pd.DataFrame, similarity: np.ndarray, out_dir: str = "artifacts"):
    """Pickle the cleaned DataFrame and similarity matrix."""
    os.makedirs(out_dir, exist_ok=True)

    movies_path     = os.path.join(out_dir, "movies.pkl")
    similarity_path = os.path.join(out_dir, "similarity.pkl")

    with open(movies_path, "wb") as f:
        pickle.dump(df, f)

    with open(similarity_path, "wb") as f:
        pickle.dump(similarity, f)

    print(f"✅ Artifacts saved to '{out_dir}/'")
    print(f"   movies.pkl      : {os.path.getsize(movies_path) / 1e6:.1f} MB")
    print(f"   similarity.pkl  : {os.path.getsize(similarity_path) / 1e6:.1f} MB")


# ─────────────────────────────────────────────
# 6. Standalone Recommend (for testing)
# ─────────────────────────────────────────────
def recommend(movie_name: str, df: pd.DataFrame, similarity: np.ndarray, top_n: int = 5):
    """
    Return top_n movie recommendations for a given title.

    Parameters
    ----------
    movie_name  : title of the query movie
    df          : preprocessed DataFrame
    similarity  : cosine-similarity matrix
    top_n       : number of recommendations (default 5)

    Returns
    -------
    list[str] — recommended movie titles
    """
    title_lower = movie_name.strip().lower()
    matches = df[df["title"].str.lower() == title_lower]

    if matches.empty:
        # Partial match fallback
        matches = df[df["title"].str.lower().str.contains(title_lower, na=False)]

    if matches.empty:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")

    idx = matches.index[0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)

    return [df.iloc[i]["title"] for i, _ in scores[1: top_n + 1]]


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Paths ──
    DATA_DIR    = "data"
    MOVIES_CSV  = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
    CREDITS_CSV = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

    # ── Validate files exist ──
    if not os.path.exists(MOVIES_CSV) or not os.path.exists(CREDITS_CSV):
        print("❌ Dataset not found!")
        print("   Please download from:")
        print("   https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print(f"   and place both CSVs inside the '{DATA_DIR}/' directory.")
        raise SystemExit(1)

    # ── Run pipeline ──
    df_raw    = load_data(MOVIES_CSV, CREDITS_CSV)
    df_clean  = preprocess(df_raw)
    sim       = build_similarity(df_clean)
    save_artifacts(df_clean, sim)

    # ── Quick smoke test ──
    print("\n🧪 Quick test — recommendations for 'Avatar':")
    try:
        recs = recommend("Avatar", df_clean, sim)
        for i, title in enumerate(recs, 1):
            print(f"   {i}. {title}")
    except ValueError as e:
        print(f"   {e}")

    print("\n🎬 All done! Run `streamlit run app.py` to launch the app.")
