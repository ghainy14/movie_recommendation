import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="wide")

# -----------------------------
# THEME TOGGLE
# -----------------------------
theme = st.sidebar.radio("Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#0E1117"
    card_color = "red"
    text_color = "white"
else:
    bg_color = "#FFFFFF"
    card_color = "#F5F5F5"
    text_color = "black"

# -----------------------------
# CUSTOM CSS (FIX BUTTON COLOR)
# -----------------------------
st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

h1 {{
    text-align: center;
    color: #FF4B4B;
}}

.stButton>button {{
    background-color: #FF4B4B !important;
    color: white !important;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}}

.movie-card {{
    background-color: {card_color};
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Smart Movie Recommender System")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    FactRatings = pd.read_csv("FactRatings.csv")
    DimMovie = pd.read_csv("DimMovie.csv")

    # Normalize columns (CRITICAL FIX)
    FactRatings.columns = FactRatings.columns.str.strip().str.lower()
    DimMovie.columns = DimMovie.columns.str.strip().str.lower()

    return FactRatings, DimMovie

FactRatings, DimMovie = load_data()

# -----------------------------
# BUILD MODELS (LIGHTWEIGHT)
# -----------------------------
@st.cache_data
def build_models(FactRatings, DimMovie):

    # 🔥 Reduce size to prevent crash
    DimMovie_small = DimMovie.head(800)

    FactRatings_small = FactRatings[
        FactRatings['movieid'].isin(DimMovie_small['movieid'])
    ]

    # USER MODEL
    user_movie_matrix = FactRatings_small.pivot_table(
        index='userid',
        columns='movieid',
        values='rating',
        fill_value=0
    )

    user_similarity = cosine_similarity(user_movie_matrix)

    # CONTENT MODEL
    count = CountVectorizer(token_pattern=None, tokenizer=lambda x: x.split('|'))

    genre_matrix = count.fit_transform(
        DimMovie_small['genres'].fillna("")
    )

    cosine_sim = cosine_similarity(genre_matrix)

    return user_movie_matrix, user_similarity, cosine_sim, DimMovie_small


user_movie_matrix, user_similarity, cosine_sim, DimMovie_small = build_models(FactRatings, DimMovie)

# -----------------------------
# DISPLAY FUNCTION
# -----------------------------
def display_movies(df):
    for _, row in df.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <h4>{row['title']}</h4>
            <p>⭐ Rating: {round(row.get('avgrating', 0),2)}</p>
            <p>🎭 Genre: {row['genres']}</p>
            <p>📅 Year: {row.get('year','N/A')}</p>
            <p>📺 Platform: {row.get('platform','N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# RECOMMEND FUNCTIONS
# -----------------------------
def recommend_movie(movie_name, top_n=5):

    matches = DimMovie_small[
        DimMovie_small['title'] == movie_name
    ]

    if matches.empty:
        return None

    idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]

    return DimMovie_small.iloc[movie_indices]


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

    return DimMovie_small[
        DimMovie_small['movieid'].isin(recommended_ids)
    ]


def top_movies(top_n=5):
    return DimMovie_small.sort_values(by='avgrating', ascending=False).head(top_n)

# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
st.sidebar.header("Options")

choice = st.sidebar.radio(
    "Recommendation Type",
    ["Movie Based", "User Based", "Top Rated"]
)

top_n = st.sidebar.slider("Number of movies", 3, 15, 5)

# -----------------------------
# UI
# -----------------------------

# 🎬 MOVIE BASED (DROPDOWN FIXED)
if choice == "Movie Based":

    movie_list = DimMovie_small['title'].dropna().unique()

    selected_movie = st.selectbox("Select a movie", movie_list)

    if st.button("Recommend Movies"):
        results = recommend_movie(selected_movie, top_n)

        if results is None:
            st.error("Movie not found")
        else:
            st.subheader("🎥 Similar Movies")
            display_movies(results)

# 👤 USER BASED
elif choice == "User Based":

    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("Recommend For User"):
        results = recommend_user(user_id, top_n)

        if results is None:
            st.error("User not found")
        else:
            st.subheader("👤 Recommended Movies")
            display_movies(results)

# ⭐ TOP MOVIES
else:
    if st.button("Show Top Movies"):
        results = top_movies(top_n)
        st.subheader("⭐ Top Rated Movies")
        display_movies(results)
