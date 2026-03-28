import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="wide")

# -----------------------------
# THEME SWITCH
# -----------------------------
theme = st.sidebar.radio("Choose Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#0E1117"
    card_color = "#262730"
    text_color = "white"
else:
    bg_color = "#F5F5F5"
    card_color = "#FFFFFF"
    text_color = "black"

# Apply theme
st.markdown("""
<style>
.stButton > button {
    background-color: #FF4B4B !important;
    color: #FF4B4B !important;
    border: 2px solid #FF4B4B !important;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Smart Movie Recommender System")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    FactRatings = pd.read_csv("FactRatings.csv")
    DimMovie = pd.read_csv("DimMovie.csv")

    FactRatings.columns = FactRatings.columns.str.strip().str.lower()
    DimMovie.columns = DimMovie.columns.str.strip().str.lower()

    # Limit size (prevent crash)
    DimMovie = DimMovie.head(600)
    FactRatings = FactRatings[FactRatings['movieid'].isin(DimMovie['movieid'])]

    return FactRatings, DimMovie

FactRatings, DimMovie = load_data()

# -----------------------------
# BUILD MODELS
# -----------------------------
@st.cache_data
def build_models(FactRatings, DimMovie):

    user_movie_matrix = FactRatings.pivot_table(
        index='userid',
        columns='movieid',
        values='rating',
        fill_value=0
    )

    user_similarity = cosine_similarity(user_movie_matrix)

    vectorizer = CountVectorizer(token_pattern=None, tokenizer=lambda x: x.split('|'))
    genre_matrix = vectorizer.fit_transform(DimMovie['genres'].fillna(""))

    cosine_sim = cosine_similarity(genre_matrix)

    return user_movie_matrix, user_similarity, cosine_sim

user_movie_matrix, user_similarity, cosine_sim = build_models(FactRatings, DimMovie)

# -----------------------------
# DISPLAY
# -----------------------------
def display_movies(df):
    for _, row in df.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <h4>{row['title']}</h4>
            <p><b>🎭 Genre:</b> {row['genres']}</p>
            <p><b>📅 Year:</b> {row['year']}</p>
            <p><b>📺 Platform:</b> {row['platform']}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# FUNCTIONS
# -----------------------------
def recommend_movie(movie_name, top_n):
    matches = DimMovie[DimMovie['title'] == movie_name]

    if matches.empty:
        return None

    idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    indices = [i[0] for i in sim_scores]
    return DimMovie.iloc[indices]


def recommend_user(user_id, top_n):
    if user_id not in user_movie_matrix.index:
        return None

    user_idx = user_movie_matrix.index.get_loc(user_id)

    sim_scores = list(enumerate(user_similarity[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    similar_users = [user_movie_matrix.index[i[0]] for i in sim_scores]

    similar_data = user_movie_matrix.loc[similar_users]
    scores = similar_data.mean().sort_values(ascending=False)

    watched = user_movie_matrix.loc[user_id]
    scores = scores[watched == 0]

    recommended_ids = scores.head(top_n).index

    return DimMovie[DimMovie['movieid'].isin(recommended_ids)]


def top_movies(top_n):
    stats = FactRatings.groupby('movieid')['rating'].mean().reset_index()
    stats.columns = ['movieid', 'avgrating']

    merged = DimMovie.merge(stats, on='movieid', how='left')
    merged['avgrating'] = merged['avgrating'].fillna(0)

    return merged.sort_values(by='avgrating', ascending=False).head(top_n)

# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
st.sidebar.header("Options")

choice = st.sidebar.radio(
    "Recommendation Type",
    ["Movie Based", "User Based", "Top Rated"]
)

top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

# -----------------------------
# UI LOGIC
# -----------------------------

# 🎬 MOVIE DROPDOWN
if choice == "Movie Based":

    movie_list = sorted(DimMovie['title'].dropna().unique())
    selected_movie = st.selectbox("Select a movie", movie_list)

    if st.button("Recommend"):
        results = recommend_movie(selected_movie, top_n)

        if results is None:
            st.error("Movie not found")
        else:
            st.subheader("🎥 Similar Movies")
            display_movies(results)

# 👤 USER
elif choice == "User Based":

    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("Recommend"):
        results = recommend_user(user_id, top_n)

        if results is None:
            st.error("User not found")
        else:
            st.subheader("👤 Recommended for You")
            display_movies(results)

# ⭐ TOP
else:

    if st.button("Show Top Movies"):
        results = top_movies(top_n)
        st.subheader("⭐ Top Rated Movies")
        display_movies(results)
