"""
🎬 CineMatch - AI Movie Recommendation System
Uses OMDb API (Free) — http://www.omdbapi.com/apikey.aspx
"""

import streamlit as st
import pandas as pd
import pickle
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CineMatch · AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --bg:#0a0a0f;--surface:#13131a;--card:#1c1c28;--border:#2a2a3d;
    --gold:#f5c518;--crimson:#e50914;--lilac:#9b8ec4;--text:#e8e8f0;--muted:#888899;--radius:12px;
}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stHeader"]{background:transparent!important;}
.block-container{padding:2rem 3rem!important;max-width:1400px;}
.hero-header{text-align:center;padding:3rem 0 2rem;}
.hero-title{font-family:'Playfair Display',serif;font-size:clamp(2.8rem,6vw,5rem);font-weight:900;background:linear-gradient(135deg,#ffffff 0%,var(--gold) 60%,var(--crimson) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0;line-height:1.1;}
.hero-sub{font-size:1rem;color:var(--muted);letter-spacing:0.3em;text-transform:uppercase;margin-top:0.5rem;}
.hero-divider{width:80px;height:3px;background:linear-gradient(90deg,var(--crimson),var(--gold));margin:1.2rem auto 0;border-radius:99px;}
[data-testid="stSelectbox"]>div>div{background:var(--card)!important;border:1.5px solid var(--border)!important;border-radius:var(--radius)!important;color:var(--text)!important;}
div.stButton>button{background:linear-gradient(135deg,var(--crimson) 0%,#b00710 100%);color:#fff;border:none;border-radius:var(--radius);font-family:'DM Sans',sans-serif;font-weight:500;font-size:1rem;padding:0.7rem 2.2rem;width:100%;transition:transform 0.15s,box-shadow 0.15s;box-shadow:0 4px 20px rgba(229,9,20,0.35);}
div.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(229,9,20,0.5);}
.section-label{font-size:0.72rem;font-weight:500;letter-spacing:0.18em;text-transform:uppercase;color:var(--muted);margin-bottom:0.4rem;}
.movie-card{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;transition:transform 0.2s,box-shadow 0.2s,border-color 0.2s;height:100%;}
.movie-card:hover{transform:translateY(-6px);box-shadow:0 20px 40px rgba(0,0,0,0.6);border-color:var(--gold);}
.movie-card img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;}
.movie-card-body{padding:1rem;}
.movie-title{font-family:'Playfair Display',serif;font-size:0.95rem;font-weight:700;color:var(--text);margin:0 0 0.3rem;line-height:1.3;}
.movie-rating{display:inline-flex;align-items:center;gap:4px;font-size:0.82rem;font-weight:500;color:var(--gold);margin-bottom:0.5rem;}
.movie-overview{font-size:0.78rem;color:var(--muted);line-height:1.5;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;}
.rank-badge{display:inline-block;background:linear-gradient(135deg,var(--crimson),var(--gold));color:#fff;font-size:0.7rem;font-weight:700;padding:2px 8px;border-radius:99px;margin-bottom:0.5rem;}
.genre-tag{display:inline-block;background:rgba(155,142,196,0.15);color:var(--lilac);font-size:0.68rem;font-weight:500;padding:2px 8px;border-radius:99px;margin:2px 2px 0 0;border:1px solid rgba(155,142,196,0.25);}
.origin-card{background:linear-gradient(135deg,var(--surface) 0%,#1a1a2e 100%);border:1px solid var(--border);border-radius:16px;padding:1.5rem;display:flex;gap:1.5rem;align-items:flex-start;margin-bottom:2rem;}
.origin-poster{width:120px;flex-shrink:0;border-radius:10px;overflow:hidden;}
.origin-poster img{width:100%;display:block;border-radius:10px;}
.origin-info{flex:1;}
.origin-title{font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:900;color:#fff;margin:0 0 0.4rem;}
.origin-meta{font-size:0.82rem;color:var(--muted);margin-bottom:0.6rem;}
.origin-overview{font-size:0.85rem;color:var(--text);line-height:1.6;}
.results-heading{font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:var(--text);margin:2.5rem 0 1rem;padding-bottom:0.5rem;border-bottom:1px solid var(--border);}
.custom-error{background:rgba(229,9,20,0.1);border:1px solid rgba(229,9,20,0.4);border-radius:var(--radius);padding:1rem 1.4rem;color:#ff6b6b;font-size:0.9rem;}
.custom-info{background:rgba(245,197,24,0.08);border:1px solid rgba(245,197,24,0.3);border-radius:var(--radius);padding:0.8rem 1.2rem;color:var(--gold);font-size:0.82rem;}
footer{display:none;}#MainMenu{display:none;}
</style>
""", unsafe_allow_html=True)

# ── OMDb API ──
OMDB_KEY    = os.getenv("OMDB_API_KEY", "")
OMDB_BASE   = "http://www.omdbapi.com/"
PLACEHOLDER = "https://placehold.co/300x450/1c1c28/f5c518?text=No+Poster"

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_movie_details(title: str) -> dict:
    empty = {"poster": PLACEHOLDER, "rating": "N/A", "overview": "", "genres": [], "year": ""}
    if not OMDB_KEY:
        return empty
    try:
        r = requests.get(OMDB_BASE, params={"apikey": OMDB_KEY, "t": title, "type": "movie", "plot": "short"}, timeout=5)
        d = r.json()
        if d.get("Response") == "False":
            return empty
        poster = d.get("Poster", PLACEHOLDER)
        if not poster or poster == "N/A":
            poster = PLACEHOLDER
        genres = [g.strip() for g in d.get("Genre", "").split(",") if g.strip()][:3]
        return {"poster": poster, "rating": d.get("imdbRating", "N/A"),
                "overview": d.get("Plot", ""), "genres": genres, "year": d.get("Year", "")}
    except Exception:
        return empty

@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        movies_df  = pickle.load(open("artifacts/movies.pkl", "rb"))
        similarity = pickle.load(open("artifacts/similarity.pkl", "rb"))
        return movies_df, similarity
    except FileNotFoundError:
        st.error("⚠️ Run `python model.py` first to generate artifacts.")
        st.stop()

def recommend(movie_name, movies_df, similarity):
    tl = movie_name.strip().lower()
    ml = movies_df["title"].str.lower()
    matches = movies_df[ml == tl]
    if matches.empty:
        matches = movies_df[ml.str.contains(tl, na=False)]
    if matches.empty:
        raise ValueError(f"Movie '{movie_name}' not found.")
    idx = matches.index[0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    results = []
    for i, _ in scores[1:]:
        if len(results) == 5: break
        results.append({"title": movies_df.iloc[i]["title"]})
    return results

def render_movie_card(title, rank):
    d = fetch_movie_details(title)
    genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in d["genres"])
    rating_str  = f"⭐ {d['rating']}" if d["rating"] != "N/A" else ""
    year_str    = f" · {d['year']}" if d["year"] else ""
    st.markdown(f"""
    <div class="movie-card">
        <img src="{d['poster']}" alt="{title}" onerror="this.src='{PLACEHOLDER}'"/>
        <div class="movie-card-body">
            <div class="rank-badge">#{rank}</div>
            <div class="movie-title">{title}</div>
            <div class="movie-rating">{rating_str}{year_str}</div>
            <div>{genres_html}</div>
            <div class="movie-overview" style="margin-top:0.5rem">{d['overview']}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def render_origin_card(title):
    d = fetch_movie_details(title)
    genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in d["genres"])
    rating_str  = f"⭐ {d['rating']}/10" if d["rating"] != "N/A" else ""
    st.markdown(f"""
    <div class="origin-card">
        <div class="origin-poster"><img src="{d['poster']}" alt="{title}" onerror="this.src='{PLACEHOLDER}'"/></div>
        <div class="origin-info">
            <div class="section-label">Because you searched</div>
            <div class="origin-title">{title}</div>
            <div class="origin-meta">{d['year']} &nbsp;|&nbsp; {rating_str}</div>
            <div style="margin-bottom:0.5rem">{genres_html}</div>
            <div class="origin-overview">{d['overview']}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def main():
    movies_df, similarity = load_artifacts()
    all_titles = sorted(movies_df["title"].tolist())

    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">CineMatch</h1>
        <p class="hero-sub">AI-Powered Movie Recommendations</p>
        <div class="hero-divider"></div>
    </div>""", unsafe_allow_html=True)

    if not OMDB_KEY:
        st.markdown("""<div class="custom-info">
        💡 <b>Tip:</b> Add <code>OMDB_API_KEY</code> to <code>.env</code> file for posters & ratings.
        Free key: <a href="http://www.omdbapi.com/apikey.aspx" target="_blank" style="color:inherit">omdbapi.com</a>
        </div>""", unsafe_allow_html=True)
        st.markdown("")

    col_search, col_btn = st.columns([4, 1], gap="medium")
    with col_search:
        st.markdown('<div class="section-label">Search a movie</div>', unsafe_allow_html=True)
        selected = st.selectbox("Movie search", [""] + all_titles, index=0, label_visibility="collapsed")
    with col_btn:
        st.markdown('<div class="section-label">&nbsp;</div>', unsafe_allow_html=True)
        clicked = st.button("🎬 Recommend", use_container_width=True)

    if clicked:
        if not selected:
            st.markdown('<div class="custom-error">⚠️ Please select a movie first.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Finding your perfect matches…"):
                try:
                    results = recommend(selected, movies_df, similarity)
                    render_origin_card(selected)
                    st.markdown('<div class="results-heading">Top 5 Recommendations</div>', unsafe_allow_html=True)
                    cols = st.columns(5, gap="medium")
                    for i, (col, movie) in enumerate(zip(cols, results)):
                        with col:
                            render_movie_card(movie["title"], i + 1)
                except ValueError as e:
                    st.markdown(f'<div class="custom-error">🎬 {e}</div>', unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;color:#444;font-size:0.75rem;padding:2rem 0 1rem;border-top:1px solid #1e1e2e;margin-top:3rem">
        CineMatch · Powered by Scikit-learn & OMDb API · Built with Streamlit
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
