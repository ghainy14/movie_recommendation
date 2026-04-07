import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
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
    text_color = "red"
else:
    bg_color = "#F5F5F5"
    card_color = "#FFFFFF"
    text_color = "black"

# Apply theme
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .movie-card {{
        background-color: {card_color};
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: {text_color};
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
    }}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def assign_platform(genre):
    if pd.isna(genre):
        return 'Netflix'
    elif 'Animation' in genre or 'Children' in genre:
        return 'Disney+'
    elif 'Action' in genre or 'Thriller' in genre:
        return 'Amazon Prime'
    elif 'Comedy' in genre:
        return 'Netflix'
    else:
        return 'Hulu'

# -----------------------------
# NAVIGATION
# -----------------------------
view = st.sidebar.radio("Select View", ["User Recommendation", "Admin ETL Dashboard"])

# =========================================================
# USER RECOMMENDATION VIEW
# =========================================================
if view == "User Recommendation":
    st.title("🎬 Smart Movie Recommender System")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    @st.cache_data
    def load_data():
        fact_ratings = pd.read_csv("FactRatings.csv")
        dim_movie = pd.read_csv("DimMovie.csv")

        fact_ratings.columns = fact_ratings.columns.str.strip().str.lower()
        dim_movie.columns = dim_movie.columns.str.strip().str.lower()

        # Limit size (prevent crash)
        dim_movie = dim_movie.head(600)
        fact_ratings = fact_ratings[fact_ratings['movieid'].isin(dim_movie['movieid'])]

        return fact_ratings, dim_movie

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
    st.sidebar.header("Recommendation Options")

    choice = st.sidebar.radio(
        "Recommendation Type",
        ["Movie Based", "User Based", "Top Rated"]
    )

    top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

    # -----------------------------
    # UI LOGIC
    # -----------------------------
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

    elif choice == "User Based":
        user_id = st.number_input("Enter User ID", min_value=1, step=1)

        if st.button("Recommend"):
            results = recommend_user(user_id, top_n)

            if results is None:
                st.error("User not found")
            else:
                st.subheader("👤 Recommended for You")
                display_movies(results)

    else:
        if st.button("Show Top Movies"):
            results = top_movies(top_n)
            st.subheader("⭐ Top Rated Movies")
            display_movies(results)

# =========================================================
# ADMIN ETL DASHBOARD
# =========================================================
elif view == "Admin ETL Dashboard":
    st.title("🛠️ Admin ETL Dashboard")
    st.write("This panel demonstrates the ETL process: Extract, Transform, and Load.")

    # -----------------------------
    # EXTRACT
    # -----------------------------
    st.header("1. Extract")

    try:
        movies = pd.read_csv("movies.csv")
        ratings = pd.read_csv("ratings.csv")
        users = pd.read_csv("users.csv")

        st.success("Raw datasets loaded successfully.")
        st.write(f"Movies shape: {movies.shape}")
        st.write(f"Ratings shape: {ratings.shape}")
        st.write(f"Users shape: {users.shape}")

        with st.expander("Preview Raw Movies Data"):
            st.dataframe(movies.head())

        with st.expander("Preview Raw Ratings Data"):
            st.dataframe(ratings.head())

        with st.expander("Preview Raw Users Data"):
            st.dataframe(users.head())

    except Exception as e:
        st.error(f"Error loading raw data files: {e}")
        st.stop()

    # -----------------------------
    # TRANSFORM
    # -----------------------------
    st.header("2. Transform")

    try:
        # Movies transformation
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
        movies['platform'] = movies['genres'].apply(assign_platform)

        # Ratings transformation
        ratings['timestamp'] = pd.to_numeric(ratings['timestamp'], errors='coerce')
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s', errors='coerce')
        ratings['hour'] = ratings['datetime'].dt.hour
        ratings['day'] = ratings['datetime'].dt.day
        ratings['month'] = ratings['datetime'].dt.month
        ratings['year'] = ratings['datetime'].dt.year

        st.success("Transformation completed successfully.")

        with st.expander("Preview Transformed Movies Data"):
            st.dataframe(movies[['movieId', 'title', 'genres', 'year', 'platform']].head())

        with st.expander("Preview Transformed Ratings Data"):
            st.dataframe(
                ratings[['userId', 'movieId', 'rating', 'timestamp', 'datetime', 'hour', 'day', 'month', 'year']].head()
            )

    except Exception as e:
        st.error(f"Error during transformation: {e}")
        st.stop()

    # -----------------------------
    # LOAD
    # -----------------------------
    st.header("3. Load")

    try:
        DimMovie = movies[['movieId', 'title', 'genres', 'year', 'platform']].drop_duplicates()
        DimUser = users[['userId', 'age', 'gender', 'occupation', 'zip']].drop_duplicates()

        DimTime = ratings[['timestamp', 'datetime', 'hour', 'day', 'month', 'year']].drop_duplicates().reset_index(drop=True)
        DimTime['timeId'] = DimTime.index + 1
        DimTime = DimTime[['timeId', 'timestamp', 'datetime', 'hour', 'day', 'month', 'year']]

        FactRatings = ratings.merge(
            DimTime[['timeId', 'timestamp']],
            on='timestamp',
            how='left'
        )[['userId', 'movieId', 'timeId', 'rating']]

        # Save warehouse CSV files
        DimMovie.to_csv("DimMovie.csv", index=False)
        DimUser.to_csv("DimUser.csv", index=False)
        DimTime.to_csv("DimTime.csv", index=False)
        FactRatings.to_csv("FactRatings.csv", index=False)

        # Save to SQLite
        engine = create_engine("sqlite:///movie_warehouse.db")
        DimMovie.to_sql("DimMovie", engine, if_exists="replace", index=False)
        DimUser.to_sql("DimUser", engine, if_exists="replace", index=False)
        DimTime.to_sql("DimTime", engine, if_exists="replace", index=False)
        FactRatings.to_sql("FactRatings", engine, if_exists="replace", index=False)

        st.success("Warehouse tables loaded successfully into CSV files and SQLite database.")

        st.subheader("Warehouse Table Shapes")
        st.write(f"DimMovie: {DimMovie.shape}")
        st.write(f"DimUser: {DimUser.shape}")
        st.write(f"DimTime: {DimTime.shape}")
        st.write(f"FactRatings: {FactRatings.shape}")

        with st.expander("Preview DimMovie"):
            st.dataframe(DimMovie.head())

        with st.expander("Preview DimUser"):
            st.dataframe(DimUser.head())

        with st.expander("Preview DimTime"):
            st.dataframe(DimTime.head())

        with st.expander("Preview FactRatings"):
            st.dataframe(FactRatings.head())

        st.success("✅ ETL process completed successfully.")

    except Exception as e:
        st.error(f"Error during loading phase: {e}")