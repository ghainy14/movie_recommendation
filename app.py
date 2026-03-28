import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Movie Recommender", layout="wide")

st.title("🎬 Smart Movie Recommender System")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    DimMovie = pd.read_csv("DimMovie.csv")
    FactRatings = pd.read_csv("FactRatings.csv")

    # Normalize column names (VERY IMPORTANT)
    DimMovie.columns = DimMovie.columns.str.strip()
    FactRatings.columns = FactRatings.columns.str.strip()

    return DimMovie, FactRatings

DimMovie, FactRatings = load_data()

# =========================
# BUILD MODELS
# =========================

@st.cache_data
def build_content_model(DimMovie):
    count = CountVectorizer(token_pattern=r'[^|]+')
    genre_matrix = count.fit_transform(DimMovie['genres'])
    cosine_sim = cosine_similarity(genre_matrix)
    return cosine_sim

cosine_sim = build_content_model(DimMovie)


@st.cache_data
def build_user_model(FactRatings):
    matrix = FactRatings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    similarity = cosine_similarity(matrix)
    return matrix, similarity

user_movie_matrix, user_similarity = build_user_model(FactRatings)

# =========================
# FUNCTIONS
# =========================

def recommend_by_movie(movie_title, top_n):
    matches = DimMovie[DimMovie['title'].str.lower() == movie_title.lower()]

    if matches.empty:
        return None

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return DimMovie.iloc[movie_indices]


def recommend_by_user(user_id, top_n):
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

    return DimMovie[DimMovie['movieId'].isin(recommended_ids)]


def top_movies(top_n):
    return DimMovie.sort_values(by='AvgRating', ascending=False).head(top_n)


# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Options")

mode = st.sidebar.radio(
    "Choose Recommendation Type",
    ["Top Movies", "By Movie", "By User"]
)

top_n = st.sidebar.slider("Number of recommendations", 1, 20, 5)

# =========================
# MAIN DISPLAY
# =========================

def display_movies(df):
    for _, row in df.iterrows():
        st.markdown(f"""
        <div style="
            background-color:#262730;
            padding:10px;
            border-radius:10px;
            margin-bottom:10px;">
            
            <h4>🎬 {row['title']}</h4>
            <p>⭐ Rating: {round(row.get('AvgRating',0),2)}</p>
            <p>🎭 Genre: {row['genres']}</p>
            <p>📺 Platform: {row.get('platform','N/A')}</p>
        </div>
        """, unsafe_allow_html=True)


# =========================
# TOP MOVIES
# =========================
if mode == "Top Movies":
    st.subheader("⭐ Top Rated Movies")
    results = top_movies(top_n)
    display_movies(results)

# =========================
# MOVIE-BASED
# =========================
elif mode == "By Movie":
    movie_name = st.text_input("Enter Movie Name")

    if st.button("Recommend"):
        results = recommend_by_movie(movie_name, top_n)

        if results is None:
            st.error("Movie not found ❌")
        else:
            st.subheader("🎥 Similar Movies")
            display_movies(results)

# =========================
# USER-BASED
# =========================
elif mode == "By User":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("Recommend"):
        results = recommend_by_user(user_id, top_n)

        if results is None:
            st.error("User not found ❌")
        else:
            st.subheader("👤 Recommended for You")
            display_movies(results)
