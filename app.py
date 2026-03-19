import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")

# =========================
# LOAD DATA (LIGHT ONLY)
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv").head(2000)
    ratings = pd.read_csv("ratings.csv").head(30000)
    return movies, ratings

movies, ratings = load_data()

# =========================
# UI ONLY FIRST
# =========================
user_id = st.sidebar.number_input(
    "Enter User ID",
    min_value=1,
    max_value=int(ratings['userId'].max()),
    value=1
)

top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

# =========================
# HEAVY COMPUTATION (ONLY WHEN BUTTON CLICKED)
# =========================
if st.button("🎯 Get Recommendations"):

    st.info("Generating recommendations... please wait ⏳")

    # -------- USER CLUSTERING --------
    sample = ratings.sample(10000, random_state=42)

    user_movie = sample.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    user_clusters = kmeans.fit_predict(user_movie)

    # -------- CONTENT SIMILARITY --------
    subset = movies.head(1000)
    genre_dummies = subset['genres'].str.get_dummies(sep='|')

    similarity = cosine_similarity(genre_dummies)
    similarity_df = pd.DataFrame(similarity, index=subset['movieId'], columns=subset['movieId'])

    # -------- RECOMMENDATION --------
    user_data = ratings[ratings['userId'] == user_id]

    if user_data.empty:
        recs = movies.head(top_n)
    else:
        top_movie = user_data.sort_values('rating', ascending=False).iloc[0]['movieId']

        if top_movie in similarity_df.columns:
            sim_scores = similarity_df[top_movie].sort_values(ascending=False)
            sim_scores = sim_scores.drop(user_data['movieId'].values, errors='ignore')
            top_ids = sim_scores.head(top_n).index
            recs = movies[movies['movieId'].isin(top_ids)]
        else:
            recs = movies.head(top_n)

    # -------- DISPLAY --------
    st.subheader("🎬 Recommended Movies")

    for _, row in recs.iterrows():
        st.write(f"🎥 {row['title']}")
