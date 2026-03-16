import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommender System")

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Create user-movie matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values
user_movie_matrix = user_movie_matrix.fillna(0)

# Compute similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity,
                                   index=user_movie_matrix.columns,
                                   columns=user_movie_matrix.columns)

# Movie selection
movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie you like:", movie_list)

# Recommendation function
def recommend_movies(movie_name, num_recommendations=5):
    similar_scores = movie_similarity_df[movie_name].sort_values(ascending=False)
    recommendations = similar_scores.iloc[1:num_recommendations+1]
    return recommendations.index

# Show recommendations
if st.button("Recommend Movies"):
    recommendations = recommend_movies(selected_movie)

    st.subheader("Recommended Movies For You")

    for movie in recommendations:
        st.write(movie)
