import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Movie Recommender", page_icon="🎬", layout="wide")

# =========================
# CUSTOM UI
# =========================
st.markdown("""
<style>
.stApp {background-color: #0E1117; color: white;}
h1 {color: #FF4B4B; text-align: center;}
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.movie-card {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Smart Movie Recommender System")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    DimMovie = pd.read_csv("DimMovie.csv")
    DimUser = pd.read_csv("DimUser.csv")
    FactRatings = pd.read_csv("FactRatings.csv")
    return DimMovie, DimUser, FactRatings

DimMovie, DimUser, FactRatings = load_data()

# =========================
# BUILD MODELS
# =========================

# 🎬 Content-Based Model
@st.cache_data
def build_content_model(DimMovie):
    count = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = count.fit_transform(DimMovie['genres'])
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    return cosine_sim

cosine_sim = build_content_model(DimMovie)

# 👤 Collaborative Model (FIXED)
@st.cache_data
def build_user_model(FactRatings):
    FactRatings.columns = FactRatings.columns.str.strip()

    matrix = FactRatings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        aggfunc='mean'
    ).fillna(0)

    similarity = cosine_similarity(matrix)

    return matrix, similarity

user_movie_matrix, user_similarity = build_user_model(FactRatings)

# =========================
# FUNCTIONS
# =========================

# 🎬 Content-based
def recommend_by_movie(movie_title, top_n):
    matches = DimMovie[DimMovie['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return []

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return DimMovie.iloc[movie_indices]


# 👤 Collaborative
def recommend_by_user(user_id, top_n):
    if user_id not in user_movie_matrix.index:
        return []

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


# ⭐ Top movies
def top_movies(top_n):
    return DimMovie.sort_values(by='AvgRating', ascending=False).head(top_n)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("Options")
option = st.sidebar.radio(
    "Choose Recommendation Type",
    ["Movie-Based", "User-Based", "Top Rated"]
)

top_n = st.sidebar.slider("Number of recommendations", 3, 15, 5)

# =========================
# MAIN UI
# =========================

# 🎬 MOVIE-BASED
if option == "Movie-Based":
    movie_name = st.text_input("Enter Movie Name")

    if st.button("Recommend Similar Movies"):
        recs = recommend_by_movie(movie_name, top_n)

        st.subheader("🎥 Recommended Movies")

        if recs.empty:
            st.warning("Movie not found!")
        else:
            for _, row in recs.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <h4>{row['title']}</h4>
                    <p>🎭 Genre: {row['genres']}</p>
                    <p>⭐ Rating: {round(row.get('AvgRating',0),2)}</p>
                    <p>📺 Platform: {row.get('platform','N/A')}</p>
                </div>
                """, unsafe_allow_html=True)


# 👤 USER-BASED
elif option == "User-Based":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("Recommend For User"):
        recs = recommend_by_user(user_id, top_n)

        st.subheader("👤 Personalized Recommendations")

        if recs.empty:
            st.warning("User not found!")
        else:
            for _, row in recs.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <h4>{row['title']}</h4>
                    <p>🎭 Genre: {row['genres']}</p>
                    <p>⭐ Rating: {round(row.get('AvgRating',0),2)}</p>
                    <p>📺 Platform: {row.get('platform','N/A')}</p>
                </div>
                """, unsafe_allow_html=True)


# ⭐ TOP MOVIES
elif option == "Top Rated":
    if st.button("Show Top Movies"):
        recs = top_movies(top_n)

        st.subheader("⭐ Top Rated Movies")

        for _, row in recs.iterrows():
            st.markdown(f"""
            <div class="movie-card">
                <h4>{row['title']}</h4>
                <p>🎭 Genre: {row['genres']}</p>
                <p>⭐ Rating: {round(row['AvgRating'],2)}</p>
                <p>📺 Platform: {row.get('platform','N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
