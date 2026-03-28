import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="wide")
st.title("🎬 Smart Movie Recommender System")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    FactRatings = pd.read_csv("FactRatings.csv")
    DimMovie = pd.read_csv("DimMovie.csv")

    # Normalize columns
    FactRatings.columns = FactRatings.columns.str.strip().str.lower()
    DimMovie.columns = DimMovie.columns.str.strip().str.lower()

    # 🔥 LIMIT SIZE (prevents crash)
    DimMovie = DimMovie.head(800)
    FactRatings = FactRatings[FactRatings['movieid'].isin(DimMovie['movieid'])]

    return FactRatings, DimMovie

FactRatings, DimMovie = load_data()

# -----------------------------
# BUILD MODELS
# -----------------------------
@st.cache_data
def build_models(FactRatings, DimMovie):

    # USER MATRIX
    user_movie_matrix = FactRatings.pivot_table(
        index='userid',
        columns='movieid',
        values='rating',
        fill_value=0
    )

    user_similarity = cosine_similarity(user_movie_matrix)

    # CONTENT MODEL
    vectorizer = CountVectorizer(token_pattern=None, tokenizer=lambda x: x.split('|'))
    genre_matrix = vectorizer.fit_transform(DimMovie['genres'].fillna(""))

    cosine_sim = cosine_similarity(genre_matrix)

    return user_movie_matrix, user_similarity, cosine_sim

# 🔥 CALL MODEL (YOU MISSED THIS BEFORE)
user_movie_matrix, user_similarity, cosine_sim = build_models(FactRatings, DimMovie)

# -----------------------------
# DISPLAY FUNCTION
# -----------------------------
def display_movies(df):
    for _, row in df.iterrows():
        st.markdown(f"""
        <div style="background:#262730;padding:10px;border-radius:10px;margin-bottom:10px">
            <h4>{row['title']}</h4>
            <p>🎭 {row['genres']}</p>
            <p>📅 {row['year']} | 📺 {row['platform']}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# RECOMMEND BY MOVIE
# -----------------------------
def recommend_movie(movie_name, top_n):

    matches = DimMovie[DimMovie['title'].str.lower() == movie_name.lower()]

    if matches.empty:
        return None

    idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    indices = [i[0] for i in sim_scores]

    return DimMovie.iloc[indices]

# -----------------------------
# RECOMMEND BY USER (FAST FIX)
# -----------------------------
def recommend_user(user_id, top_n):

    if user_id not in user_movie_matrix.index:
        return None

    user_idx = user_movie_matrix.index.get_loc(user_id)

    # 🔥 Only top similar users (prevents crash)
    sim_scores = list(enumerate(user_similarity[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    similar_users = [user_movie_matrix.index[i[0]] for i in sim_scores]

    # Aggregate ratings from similar users
    similar_data = user_movie_matrix.loc[similar_users]

    scores = similar_data.mean().sort_values(ascending=False)

    # Remove already watched
    watched = user_movie_matrix.loc[user_id]
    scores = scores[watched == 0]

    recommended_ids = scores.head(top_n).index

    return DimMovie[DimMovie['movieid'].isin(recommended_ids)]

# -----------------------------
# TOP MOVIES
# -----------------------------
def top_movies(top_n):

    stats = FactRatings.groupby('movieid')['rating'].mean().reset_index()
    stats.columns = ['movieid', 'avgrating']

    merged = DimMovie.merge(stats, on='movieid', how='left')
    merged['avgrating'] = merged['avgrating'].fillna(0)

    return merged.sort_values(by='avgrating', ascending=False).head(top_n)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Options")
choice = st.sidebar.radio(
    "Choose Recommendation Type",
    ["Movie Based", "User Based", "Top Rated"]
)

top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

# -----------------------------
# UI
# -----------------------------

# 🎬 MOVIE
if choice == "Movie Based":
    movie_name = st.text_input("Enter movie name")

    if st.button("Recommend"):
        results = recommend_movie(movie_name, top_n)

        if results is None:
            st.error("Movie not found")
        else:
            display_movies(results)

# 👤 USER
elif choice == "User Based":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("Recommend"):
        results = recommend_user(user_id, top_n)

        if results is None:
            st.error("User not found")
        else:
            display_movies(results)

# ⭐ TOP
else:
    if st.button("Show Top Movies"):
        results = top_movies(top_n)
        display_movies(results)
