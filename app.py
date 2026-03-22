import streamlit as st
import pandas as pd
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #ff1f1f;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1c1f26;
}

/* Buttons */
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

/* Button hover */
.stButton>button:hover {
    background-color: #0E1117;
}

/* Titles */
h1, h2, h3 {
    color: #FFD700;
}

/* Cards effect */
.css-1r6slb0 {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
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
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def compute_similarity(movies):
    subset = movies.head(500)  # VERY IMPORTANT (limit size)
    genre_dummies = subset['genres'].str.get_dummies(sep='|')
    similarity = cosine_similarity(genre_dummies)
    return pd.DataFrame(similarity, index=subset['movieId'], columns=subset['movieId']), subset

similarity_df, subset_movies = compute_similarity(movies)

movie_list = subset_movies['title'].values
selected_movie = st.selectbox("Select a movie you like:", movie_list)

if st.button("Get Similar Movies"):
    movie_id = subset_movies[subset_movies['title'] == selected_movie]['movieId'].values[0]
    
    sim_scores = similarity_df[movie_id].sort_values(ascending=False).iloc[1:6]
    rec_ids = sim_scores.index
    
    recs = subset_movies[subset_movies['movieId'].isin(rec_ids)]
    
    st.subheader("Recommended Movies")
    for _, row in recs.iterrows():
        st.write(row['title'])
