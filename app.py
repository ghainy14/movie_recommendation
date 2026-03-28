import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="wide")
st.title("🎬 Smart Movie Recommender System")

# -----------------------------
# LOAD DATA (SAFE)
# -----------------------------
@st.cache_data
def load_data():
    FactRatings = pd.read_csv("FactRatings.csv")
    DimMovie = pd.read_csv("DimMovie.csv")
    DimUser = pd.read_csv("DimUser.csv")

    # 🔥 FIX: normalize column names
    FactRatings.columns = FactRatings.columns.str.strip().str.lower()
    DimMovie.columns = DimMovie.columns.str.strip().str.lower()
    DimUser.columns = DimUser.columns.str.strip().str.lower()

    return FactRatings, DimMovie, DimUser

FactRatings, DimMovie, DimUser = load_data()

# -----------------------------
# BUILD MODELS
# -----------------------------
@st.cache_data
def build_models(FactRatings, DimMovie):

    # USER-ITEM MATRIX (Collaborative)
    user_movie_matrix = FactRatings.pivot_table(
        index='userid',
        columns='movieid',
        values='rating',
        fill_value=0
    )

    user_similarity = cosine_similarity(user_movie_matrix)

    # CONTENT-BASED (Genres)
    count = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = count.fit_transform(DimMovie['genres'].fillna(""))

    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

    return user_movie_matrix, user_similarity, cosine_sim

user_movie_matrix, user_similarity, cosine_sim = build_models(FactRatings, DimMovie)

# -----------------------------
# HELPER: DISPLAY MOVIES
# -----------------------------
def display_movies(df):
    for _, row in df.iterrows():
        st.markdown(f"""
        <div style="background:#262730;padding:10px;border-radius:10px;margin-bottom:10px">
            <h4>{row['title']}</h4>
            <p>⭐ Rating: {round(row.get('avgrating', 0),2)}</p>
            <p>🎭 Genre: {row['genres']}</p>
            <p>📅 Year: {row['year']}</p>
            <p>📺 Platform: {row['platform']}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# RECOMMENDATION FUNCTIONS
# -----------------------------

# 🎬 CONTENT-BASED
def recommend_movie(movie_name, top_n=5):
    matches = DimMovie[DimMovie['title'].str.lower() == movie_name.lower()]

    if matches.empty:
        return None

    idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return DimMovie.iloc[movie_indices]


# 👤 COLLABORATIVE
def recommend_user(user_id, top_n=5):

    if user_id not in user_movie_matrix.index:
        return None

    user_idx = list(user_movie_matrix.index).index(user_id)
    sim_scores = user_similarity[user_idx]

    user_ratings = user_movie_matrix.iloc[user_idx]
    unrated_movies = user_ratings[user_ratings == 0].index

    scores = {
        movie: sim_scores @ user_movie_matrix[movie] / (sim_scores.sum() + 1e-8)
        for movie in unrated_movies
    }

    recommended_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]

    return DimMovie[DimMovie['movieid'].isin(recommended_ids)]


# ⭐ TOP MOVIES
def top_movies(top_n=5):
    return DimMovie.sort_values(by='avgrating', ascending=False).head(top_n)


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Options")
choice = st.sidebar.radio(
    "Choose Recommendation Type",
    ["Movie Based", "User Based", "Top Rated"]
)

top_n = st.sidebar.slider("Number of recommendations", 3, 15, 5)

# -----------------------------
# UI LOGIC
# -----------------------------

# 🎬 MOVIE BASED
if choice == "Movie Based":
    movie_name = st.text_input("Enter movie name")

    if st.button("Recommend"):
        results = recommend_movie(movie_name, top_n)

        if results is None:
            st.error("Movie not found!")
        else:
            st.subheader("🎥 Similar Movies")
            display_movies(results)

# 👤 USER BASED
elif choice == "User Based":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("Recommend"):
        results = recommend_user(user_id, top_n)

        if results is None:
            st.error("User not found!")
        else:
            st.subheader("👤 Recommended for You")
            display_movies(results)

# ⭐ TOP MOVIES
elif choice == "Top Rated":
    if st.button("Show Top Movies"):
        results = top_movies(top_n)
        st.subheader("⭐ Top Rated Movies")
        display_movies(results)
