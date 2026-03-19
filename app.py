import streamlit as st
import pandas as pd

st.title("🎬 Movie Recommender System")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv", nrows=1000)
    ratings = pd.read_csv("ratings.csv", nrows=5000)
    
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean','count']).reset_index()
    movie_stats.rename(columns={'mean':'AvgRating', 'count':'TotalRatings'}, inplace=True)
    
    movies = movies.merge(movie_stats, on='movieId', how='left')
    movies['AvgRating'] = movies['AvgRating'].fillna(0)
    
    return movies

movies = load_data()

st.sidebar.title("Options")
top_n = st.sidebar.slider("Number of movies", 3, 10, 5)

if st.button("Recommend Top Movies"):
    top_movies = movies.sort_values(by='AvgRating', ascending=False).head(top_n)
    
    for _, row in top_movies.iterrows():
        st.write(f"🎬 {row['title']} ({round(row['AvgRating'],2)})")
