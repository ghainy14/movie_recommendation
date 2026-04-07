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
    text_color = "red"
else:
    bg_color = "#F5F5F5"
    card_color = "#262730"
    text_color = "black"


st.markdown(f"""
<style>
/* App background */
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

/* Movie cards */
.movie-card {{
    background-color: {card_color};
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: {text_color};
    box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
}}

/* Labels (like movieid, title, etc.) */
label, .stTextInput label, .stSelectbox label, .stNumberInput label {{
    color: {text_color} !important;
    font-weight: 600;
}}

/* Input fields */
input, textarea {{
    color: {text_color} !important;
    background-color: {card_color} !important;
}}

/* Dropdown (selectbox) */
div[data-baseweb="select"] > div {{
    background-color: {card_color} !important;
    color: {text_color} !important;
}}

/* Dropdown text */
div[data-baseweb="select"] span {{
    color: {text_color} !important;
}}

/* Buttons */
.stButton button {{
    color: {text_color};
    background-color: {card_color};
    border: 1px solid gray;
}}

/* Sidebar text */
section[data-testid="stSidebar"] * {{
    color: {text_color} !important;
}}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    FactRatings = pd.read_csv("FactRatings.csv")
    DimMovie = pd.read_csv("DimMovie.csv")

    FactRatings.columns = FactRatings.columns.str.strip().str.lower()
    DimMovie.columns = DimMovie.columns.str.strip().str.lower()

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
# DISPLAY FUNCTION
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
# RECOMMENDATION FUNCTIONS
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
# MODE SWITCH
# -----------------------------
mode = st.sidebar.radio("Mode", ["User", "Admin"])

# =============================
# USER MODE
# =============================
if mode == "User":

    st.sidebar.header("Options")

    choice = st.sidebar.radio(
        "Recommendation Type",
        ["Movie Based", "User Based", "Top Rated"]
    )

    top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

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

# =============================
# ADMIN MODE (ETL)
# =============================
else:

    st.header("🛠️ Admin ETL Panel")

    etl_option = st.selectbox(
        "Choose Operation",
        ["Extract", "Transform", "Load"]
    )

    # -------------------------
    # EXTRACT
    # -------------------------
    if etl_option == "Extract":

        st.subheader("📤 Extract Data")

        table = st.selectbox("Select Table", ["DimMovie", "FactRatings"])
        df = DimMovie if table == "DimMovie" else FactRatings

        st.dataframe(df)

        query = st.text_input("Optional Query (e.g. year > 2015)")

        if query:
            try:
                filtered = df.query(query)
                st.dataframe(filtered)
            except:
                st.error("Invalid query")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "data.csv", "text/csv")

    # -------------------------
    # TRANSFORM
    # -------------------------
    elif etl_option == "Transform":

        st.subheader("🔄 Transform Data")

        table = st.selectbox("Select Table", ["DimMovie", "FactRatings"])
        df = DimMovie if table == "DimMovie" else FactRatings

        column = st.selectbox("Column", df.columns)
        condition = st.text_input("Condition (e.g. year < 2000)")
        new_value = st.text_input("New Value")

        if st.button("Apply Transformation"):
            try:
                df.loc[df.query(condition).index, column] = new_value
                st.success("Transformation Applied")
                st.dataframe(df.head())
            except:
                st.error("Error in transformation")

    # -------------------------
    # LOAD
    # -------------------------
    elif etl_option == "Load":

        st.subheader("📥 Load Data")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            new_data = pd.read_csv(uploaded_file)
            st.dataframe(new_data.head())

            table = st.selectbox("Insert into", ["DimMovie", "FactRatings"])

            if st.button("Insert Data"):
                if table == "DimMovie":
                    DimMovie = pd.concat([DimMovie, new_data], ignore_index=True)
                else:
                    FactRatings = pd.concat([FactRatings, new_data], ignore_index=True)

                st.success("Data Loaded Successfully")

        st.subheader("Manual Insert")

        new_row = {}
        for col in DimMovie.columns:
            new_row[col] = st.text_input(f"{col}")

        if st.button("Insert Row"):
            DimMovie = pd.concat([DimMovie, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Row Inserted")
