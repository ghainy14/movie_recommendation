import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")

# =========================
# LOAD DATA (REDUCED SIZE)
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv").head(3000)
    ratings = pd.read_csv("ratings.csv").head(50000)

    # Extract year
    movies['Year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

    # Movie stats
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean','count']).reset_index()
    movie_stats.rename(columns={'mean':'AvgRating', 'count':'TotalRatings'}, inplace=True)

    movies = movies.merge(movie_stats, on='movieId', how='left')
    movies['AvgRating'] = movies['AvgRating'].fillna(0)

    return movies, ratings

movies, ratings = load_data()

# =========================
# USER CLUSTERING (SAMPLED)
# =========================
@st.cache_data
def cluster_users(ratings):
    sample = ratings.sample(20000, random_state=42)

    user_movie = sample.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(user_movie)

    return pd.DataFrame({
        'userId': user_movie.index,
        'Cluster': clusters
    })

user_clusters = cluster_users(ratings)

# =========================
# CONTENT SIMILARITY (LIMITED)
# =========================
@st.cache_data
def compute_similarity(movies):
    subset = movies.head(2000)
    genre_dummies = subset['genres'].str.get_dummies(sep='|')

    similarity = cosine_similarity(genre_dummies)

    return pd.DataFrame(similarity, index=subset['movieId'], columns=subset['movieId'])

similarity_df = compute_similarity(movies)

# =========================
# RECOMMENDER
# =========================
def recommend(user_id, top_n=5):
    user_data = ratings[ratings['userId'] == user_id]

    if user_data.empty:
        return movies.sort_values(by='AvgRating', ascending=False).head(top_n)

    top_movie = user_data.sort_values('rating', ascending=False).iloc[0]['movieId']

    if top_movie not in similarity_df.columns:
        return movies.sort_values(by='AvgRating', ascending=False).head(top_n)

    sim_scores = similarity_df[top_movie].sort_values(ascending=False)

    sim_scores = sim_scores.drop(user_data['movieId'].values, errors='ignore')

    top_ids = sim_scores.head(top_n).index

    return movies[movies['movieId'].isin(top_ids)][['title', 'genres', 'AvgRating']]

# =========================
# UI
# =========================
user_id = st.sidebar.number_input(
    "Enter User ID",
    min_value=1,
    max_value=int(ratings['userId'].max()),
    value=1
)

top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

if st.button("🎯 Get Recommendations"):
    recs = recommend(user_id, top_n)

    st.subheader("Recommended Movies")

    for _, row in recs.iterrows():
        st.write(f"🎬 {row['title']}")
        st.write(f"Genre: {row['genres']}")
        st.write(f"⭐ Rating: {round(row['AvgRating'],2)}")
        st.write("---")
