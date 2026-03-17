import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Create matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Compute similarity
movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

# UI
selected_movie = st.selectbox("Select a movie you like:", movies['title'].values)

def recommend_movies(movie_name, n=5):
    similar = movie_similarity_df[movie_name].sort_values(ascending=False)
    return similar.iloc[1:n+1].index

if st.button("Recommend"):
    try:
        recs = recommend_movies(selected_movie)

        st.subheader("🎯 Recommended Movies")
        for i, movie in enumerate(recs, 1):
            st.write(f"{i}. {movie}")

    except Exception as e:
        st.error("Something went wrong. Try another movie.")
