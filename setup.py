import os
import ast
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_artifacts():
    if os.path.exists("artifacts/movies.pkl") and os.path.exists("artifacts/similarity.pkl"):
        print("Artifacts already exist, skipping...")
        return

    print("Building model artifacts...")
    os.makedirs("artifacts", exist_ok=True)

    movies  = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")
    if "movie_id" in credits.columns:
        credits = credits.rename(columns={"movie_id": "id"})

    df = movies.merge(credits, on="id")
    df = df[["id", "title_x", "overview", "genres", "keywords", "cast", "crew"]]
    df = df.rename(columns={"title_x": "title"})
    df = df.dropna(subset=["overview"]).copy()

    def parse_list(obj, key="name", limit=None):
        try:
            items = ast.literal_eval(obj)
            result = [d[key].replace(" ", "") for d in items if key in d]
            return result[:limit] if limit else result
        except: return []

    def get_director(crew):
        try:
            for p in ast.literal_eval(crew):
                if p.get("job") == "Director":
                    return [p["name"].replace(" ", "")]
        except: pass
        return []

    df["genres"]   = df["genres"].apply(lambda x: parse_list(x))
    df["keywords"] = df["keywords"].apply(lambda x: parse_list(x))
    df["cast"]     = df["cast"].apply(lambda x: parse_list(x, limit=5))
    df["crew"]     = df["crew"].apply(get_director)
    df["tags"]     = (df["overview"].apply(str.split) + df["genres"] + df["keywords"] + df["cast"] + df["crew"]).apply(lambda x: " ".join(x).lower())

    df = df[["id", "title", "tags"]].reset_index(drop=True)

    cv  = CountVectorizer(max_features=5000, stop_words="english")
    vec = cv.fit_transform(df["tags"]).toarray()
    sim = cosine_similarity(vec)

    pickle.dump(df,  open("artifacts/movies.pkl", "wb"))
    pickle.dump(sim, open("artifacts/similarity.pkl", "wb"))
    print("Done! Artifacts created.")

if __name__ == "__main__":
    build_artifacts()