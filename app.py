import streamlit as st
import pandas as pd

st.title("🎬 Movie Recommender System")

# Load VERY small data
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv", nrows=500)
    ratings = pd.read_csv("ratings.csv", nrows=2000)
    return movies, ratings

movies, ratings = load_data()

st.success("Data loaded successfully ✅")

user_id = st.number_input("Enter User ID", 1, 100, 1)

if st.button("Test Recommendation"):
    st.write("App is working ✅")
