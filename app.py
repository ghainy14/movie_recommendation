import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Smart Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# STYLING
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}

h1 {
    color: #FF4B4B;
    text-align: center;
}

.movie-card {
    background-color: #262730;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Smart Movie Recommender System")

# -----------------------------
# LOAD DATA FROM DATA WAREHOUSE
# -----------------------------
@st.cache_data
def load_data():
    DimMovie = pd.read_csv("DimMovie.csv")
    DimUser = pd.read_csv("DimUser.csv")
    FactRatings = pd.read_csv("FactRatings.csv")
    DimTime = pd.read_csv("DimTime.csv")

    return DimMovie, DimUser, FactRatings, DimTime

DimMovie, DimUser, FactRatings, DimTime = load_data()

# -----------------------------
# CONTENT MODEL (GENRE + PLATFORM)
# -----------------------------
@st.cache_data
def build_content_model(DimMovie):
    DimMovie['features'] = DimMovie['genres'] + " " + DimMovie['platform']
    features = DimMovie['features'].str.get_dummies(sep=' ')
    similarity = cosine_similarity(features)
    return similarity

cosine_sim = build_content_model(DimMovie)

# -----------------------------
# USER MODEL
# -----------------------------
@st.cache_data
def build_user_model(FactRatings):
    matrix = FactRatings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    similarity = cosine_similarity(matrix)
    return matrix, similarity

user_movie_matrix, user_similarity = build_user_model(FactRatings)

# -----------------------------
# DISPLAY FUNCTION
# -----------------------------
def display_movies(df):
    for _, row in df.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <h4>🎬 {row['title']}</h4>
            <p>🎭 Genre: {row['genres']}</p>
            <p>📺 Platform: {row['platform']}</p>
            <p>⭐ Rating: {round(row['AvgRating'],2)}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Options")

option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ["Top Movies", "Movie-Based", "User-Based"]
)

top_n = st.sidebar.slider("Number of recommendations", 3, 15, 5)

# -----------------------------
# TOP MOVIES
# -----------------------------
if option == "Top Movies":
    st.subheader("⭐ Top Rated Movies")

    top_movies = DimMovie.sort_values(by='AvgRating', ascending=False).head(top_n)
    display_movies(top_movies)

# -----------------------------
# MOVIE-BASED
# -----------------------------
elif option == "Movie-Based":
    st.subheader("🎥 Recommend by Movie")

    selected_movie = st.selectbox("Select a movie:", DimMovie['title'])

    if st.button("Recommend"):
        idx = DimMovie[DimMovie['title'] == selected_movie].index[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        indices = [i[0] for i in sim_scores]
        recs = DimMovie.iloc[indices]

        display_movies(recs)

# -----------------------------
# USER-BASED (WITH TIME FILTER)
# -----------------------------
elif option == "User-Based":
    st.subheader("👤 Recommend by User")

    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    # NEW: TIME CONTEXT
    selected_hour = st.slider("Select Hour of Day (Context)", 0, 23, 12)

    if st.button("Recommend for User"):
        if user_id not in user_movie_matrix.index:
            st.error("User not found!")
        else:
            user_idx = list(user_movie_matrix.index).index(user_id)
            sim_scores = user_similarity[user_idx]

            user_ratings = user_movie_matrix.iloc[user_idx]
            unrated_movies = user_ratings[user_ratings == 0].index

            scores = {
                movie: sim_scores @ user_movie_matrix[movie] / (sim_scores.sum() + 1e-8)
                for movie in unrated_movies
            }

            recommended_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
            recs = DimMovie[DimMovie['movieId'].isin(recommended_ids)]

            st.write(f"🕒 Recommendations for hour: {selected_hour}")
            display_movies(recs)
