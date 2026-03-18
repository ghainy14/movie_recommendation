import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")
st.markdown("Hybrid Recommendation System (Clustering + Content-Based Filtering)")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # Extract year
    movies['Year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

    # Movie stats
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean','count']).reset_index()
    movie_stats.rename(columns={'mean':'AvgRating', 'count':'TotalRatings'}, inplace=True)

    movies = movies.merge(movie_stats, on='movieId', how='left')
    movies['AvgRating'] = movies['AvgRating'].fillna(0)
    movies['TotalRatings'] = movies['TotalRatings'].fillna(0)

    return movies, ratings

movies, ratings = load_data()

# =========================
# USER CLUSTERING
# =========================
@st.cache_data
def cluster_users(ratings):
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(user_movie_matrix)

    user_clusters = pd.DataFrame({
        'userId': user_movie_matrix.index,
        'Cluster': clusters
    })

    return user_clusters

user_clusters = cluster_users(ratings)

# =========================
# CONTENT-BASED SIMILARITY
# =========================
@st.cache_data
def compute_similarity(movies):
    genre_dummies = movies['genres'].str.get_dummies(sep='|')
    similarity_matrix = cosine_similarity(genre_dummies)
    return pd.DataFrame(similarity_matrix, index=movies['movieId'], columns=movies['movieId'])

similarity_df = compute_similarity(movies)

# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend(user_id, top_n=5):
    user_rated = ratings[ratings['userId'] == user_id]

    if user_rated.empty:
        return movies.sort_values(by='AvgRating', ascending=False).head(top_n)

    # Get top rated movie
    top_movie_id = user_rated.sort_values('rating', ascending=False).iloc[0]['movieId']

    # Similar movies
    sim_scores = similarity_df[top_movie_id].sort_values(ascending=False)

    # Remove watched movies
    sim_scores = sim_scores.drop(user_rated['movieId'].values, errors='ignore')

    top_ids = sim_scores.head(top_n).index

    return movies[movies['movieId'].isin(top_ids)][['title', 'genres', 'AvgRating']]

# =========================
# SIDEBAR (USER INPUT)
# =========================
st.sidebar.header("User Options")

user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=int(ratings['userId'].max()), value=1)

top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

# =========================
# MAIN ACTION
# =========================
if st.button("🎯 Get Recommendations"):
    recs = recommend(user_id, top_n)

    st.subheader(f"Top {top_n} Recommendations for User {user_id}")

    for i, row in recs.iterrows():
        st.markdown(f"**{row['title']}**")
        st.write(f"Genre: {row['genres']}")
        st.write(f"⭐ Rating: {round(row['AvgRating'],2)}")
        st.write("---")

# =========================
# OPTIONAL: SHOW CLUSTERS
# =========================
if st.checkbox("Show User Cluster Distribution"):
    st.subheader("User Clusters")
    st.bar_chart(user_clusters['Cluster'].value_counts())
