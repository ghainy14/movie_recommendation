import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}

h1 {
    color: #FF4B4B;
    text-align: center;
}

.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

section[data-testid="stSidebar"] {
    background-color: #1C1F26;
}

.movie-card {
    background-color: #262730;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender System")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # Compute stats
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean','count']).reset_index()
    movie_stats.columns = ['movieId','AvgRating','TotalRatings']

    movies = movies.merge(movie_stats, on='movieId', how='left')
    movies['AvgRating'] = movies['AvgRating'].fillna(0)

    return movies, ratings

movies, ratings = load_data()

# -----------------------------
# CONTENT-BASED MODEL
# -----------------------------
@st.cache_data
def compute_content_model(movies):
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    genre_matrix = movies['genres_clean'].str.get_dummies(sep=' ')
    similarity = cosine_similarity(genre_matrix)
    return similarity

cosine_sim = compute_content_model(movies)

# -----------------------------
# COLLABORATIVE MODEL
# -----------------------------
@st.cache_data
def compute_user_model(ratings):
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_similarity = cosine_similarity(user_movie_matrix)
    return user_movie_matrix, user_similarity

user_movie_matrix, user_similarity = compute_user_model(ratings)

# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
st.sidebar.title("⚙️ Options")

option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ["Top Movies", "Movie-Based", "User-Based"]
)

top_n = st.sidebar.slider("Number of recommendations", 3, 15, 5)

# -----------------------------
# 1. TOP MOVIES
# -----------------------------
if option == "Top Movies":
    st.subheader("⭐ Top Rated Movies")

    top_movies = movies.sort_values(by='AvgRating', ascending=False).head(top_n)

    for _, row in top_movies.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <h4>🎬 {row['title']}</h4>
            <p>⭐ Rating: {round(row['AvgRating'],2)}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# 2. MOVIE-BASED
# -----------------------------
elif option == "Movie-Based":
    st.subheader("🎥 Find Similar Movies")

    movie_list = movies['title'].values
    selected_movie = st.selectbox("Select a movie:", movie_list)

    if st.button("Recommend Similar Movies"):
        idx = movies[movies['title'] == selected_movie].index[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        movie_indices = [i[0] for i in sim_scores]
        recs = movies.iloc[movie_indices]

        for _, row in recs.iterrows():
            st.markdown(f"""
            <div class="movie-card">
                <h4>🎬 {row['title']}</h4>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# 3. USER-BASED
# -----------------------------
elif option == "User-Based":
    st.subheader("👤 Personalized Recommendations")

    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("Recommend for User"):
        if user_id not in user_movie_matrix.index:
            st.error("User not found!")
        else:
            user_idx = list(user_movie_matrix.index).index(user_id)
            sim_scores = user_similarity[user_idx]

            user_ratings = user_movie_matrix.iloc[user_idx]
            unrated_movies = user_ratings[user_ratings == 0].index

            scores = {
                movie: sim_scores @ user_movie_matrix[movie] / (sim_scores.sum() + 1e-8)
                for movie in unrated_movies
            }

            recommended_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
            recs = movies[movies['movieId'].isin(recommended_ids)]

            for _, row in recs.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <h4>🎬 {row['title']}</h4>
                </div>
                """, unsafe_allow_html=True)
