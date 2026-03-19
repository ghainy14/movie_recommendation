import streamlit as st
import pandas as pd

st.title("🎬 Movie Recommender System")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv", nrows=1000)
    ratings = pd.read_csv("ratings.csv", nrows=5000)
    return movies, ratings

movies, ratings = load_data()

st.success("Data loaded successfully ✅")
st.write(movies.head())
