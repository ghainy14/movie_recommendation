import pandas as pd

# =========================
# STEP 1: LOAD USER DATA (MovieLens 100k)
# =========================

users = pd.read_csv(
    "u.user",
    sep="|",
    names=["userId", "age", "gender", "occupation", "zip"]
)

# =========================
# STEP 2: ORIGINAL USER TABLE
# =========================

user = users.copy()

# =========================
# STEP 3: SPLIT INTO TWO DATA SOURCES
# =========================

# Split by userId (half)
half = len(users) // 2

# =========================
# STEP 4: SAVE FILES
# =========================

user.to_csv("user.csv", index=False)
user_sdb1.to_csv("user_sdb1.csv", index=False)
user_sdb2.to_csv("user_sdb2.csv", index=False)

# =========================
# STEP 5: PREVIEW
# =========================

print("Original User Table:")
print(user.head())

print("\nUser SDB1:")
print(user_sdb1.head())

print("\nUser SDB2:")
print(user_sdb2.head())
import pandas as pd
import numpy as np

# Step 1: Load movies dataset
movies = pd.read_csv("movies.csv")

# Step 2: Create platform column using rule-based logic
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

movies['platform'] = movies['genres'].apply(assign_platform)

# Step 3: Clean movie title and extract year
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['year'] = movies['year'].astype(float)

# Step 4: Create DimMovie table
dim_movie = movies[['movieId', 'title', 'genres', 'year', 'platform']].drop_duplicates()

# Step 5: Add surrogate key (movie_key)
#dim_movie = dim_movie.reset_index(drop=True)
#dim_movie['movie_key'] = dim_movie.index + 1

# Step 6: Reorder columns
dim_movie = dim_movie[['movieId', 'title', 'genres', 'year', 'platform']]

# Step 7: Save outputs
movies.to_csv("movies_updated.csv", index=False)
dim_movie.to_csv("DimMovie.csv", index=False)

# Step 8: Preview
print(dim_movie.head())
# Split users into two groups
user_ids = ratings['userId'].unique()
mid = len(user_ids) // 2

users_sdb1 = user_ids[:mid]
users_sdb2 = user_ids[mid:]

# Create two source databases
ratings_sdb1 = ratings[ratings['userId'].isin(users_sdb1)]
ratings_sdb2 = ratings[ratings['userId'].isin(users_sdb2)]

movies_sdb1 = movies.copy()
movies_sdb2 = movies.copy()

# Save SDBs
movies_sdb1.to_csv("movies_sdb1.csv", index=False)
ratings_sdb1.to_csv("ratings_sdb1.csv", index=False)

movies_sdb2.to_csv("movies_sdb2.csv", index=False)
ratings_sdb2.to_csv("ratings_sdb2.csv", index=False)

print("✅ SDB1 and SDB2 created and saved")
# Combine both SDBs
FactRatings = pd.concat([ratings_sdb1, ratings_sdb2])

# Aggregate movie stats
movie_stats = FactRatings.groupby('movieId')['rating'].agg(['mean','count']).reset_index()
movie_stats.columns = ['movieId', 'AvgRating', 'TotalRatings']

# Create DimMovie
DimMovie = movies.merge(movie_stats, on='movieId', how='left')
DimMovie['AvgRating'] = DimMovie['AvgRating'].fillna(0)
DimMovie['TotalRatings'] = DimMovie['TotalRatings'].fillna(0)

# Create DimUser
DimUser = FactRatings.groupby('userId').agg(
    AvgRating=('rating', 'mean'),
    TotalRatings=('rating', 'count')
).reset_index()

print("✅ Data Warehouse tables created")
import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("ratings.csv")  

# Step 2: Convert scientific notation timestamp → integer
df['timestamp'] = df['timestamp'].astype(float).astype(int)

# Step 3: Convert Unix timestamp → datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Step 4: Extract time attributes
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Step 5: Create DimTime table
dim_time = df[['timestamp', 'datetime', 'hour', 'day', 'month', 'year']].drop_duplicates()

# Step 6: Add surrogate key
dim_time = dim_time.reset_index(drop=True)
dim_time['time_id'] = dim_time.index + 1

# Step 7: Reorder columns
dim_time = dim_time[['time_id', 'timestamp', 'datetime', 'hour', 'day', 'month', 'year']]

dim_time.to_csv("DimTime.csv", index=False)

print(dim_time.head())
# Save CSV files
DimMovie.to_csv("DimMovie.csv", index=False)
DimUser.to_csv("DimUser.csv", index=False)
FactRatings.to_csv("FactRatings.csv", index=False)

# Save to SQLite
engine = create_engine('sqlite:///movie_warehouse.db')

DimMovie.to_sql('DimMovie', engine, if_exists='replace', index=False)
DimUser.to_sql('DimUser', engine, if_exists='replace', index=False)
FactRatings.to_sql('FactRatings', engine, if_exists='replace', index=False)

print("✅ Data Warehouse saved (CSV + SQLite)")
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# User-item matrix
user_movie_matrix = FactRatings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# Convert genres to vectors
count = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = count.fit_transform(DimMovie['genres'])

cosine_sim = cosine_similarity(genre_matrix, genre_matrix)


def content_based(movie_title, top_n=5):
    matches = DimMovie[DimMovie['title'].str.lower() == movie_title.lower()]
    
    if matches.empty:
        return f"❌ Movie '{movie_title}' not found."
    
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return DimMovie['title'].iloc[movie_indices].tolist()
user_similarity = cosine_similarity(user_movie_matrix)

def collaborative(user_id, top_n=5):
    if user_id not in user_movie_matrix.index:
        return f"❌ User {user_id} not found."
    
    user_idx = list(user_movie_matrix.index).index(user_id)
    sim_scores = user_similarity[user_idx]
    
    user_ratings = user_movie_matrix.iloc[user_idx]
    unrated_movies = user_ratings[user_ratings == 0].index
    
    scores = {
        movie: sim_scores @ user_movie_matrix[movie] / (sim_scores.sum() + 1e-8)
        for movie in unrated_movies
    }
    
    recommended_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    
    return DimMovie[DimMovie['movieId'].isin(recommended_ids)]['title'].tolist()
def top_movies(top_n=5):
    return DimMovie.sort_values(by='AvgRating', ascending=False)['title'].head(top_n).tolist()
def main():
    print("\n🎬 MOVIE RECOMMENDER SYSTEM")
    print("1. Recommend based on Movie")
    print("2. Recommend based on User ID")
    print("3. Show Top Rated Movies")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            movie_name = input("Enter movie name: ")
            n = int(input("How many recommendations? "))
            
            recs = content_based(movie_name, n)
            print("\n🎥 Similar Movies:")
            for i, movie in enumerate(recs, 1):
                print(f"{i}. {movie}")
        
        elif choice == '2':
            user_id = int(input("Enter user ID: "))
            n = int(input("How many recommendations? "))
            
            recs = collaborative(user_id, n)
            print("\n👤 Recommended for You:")
            for i, movie in enumerate(recs, 1):
                print(f"{i}. {movie}")
        
        elif choice == '3':
            n = int(input("How many top movies? "))
            
            recs = top_movies(n)
            print("\n⭐ Top Rated Movies:")
            for i, movie in enumerate(recs, 1):
                print(f"{i}. {movie}")
        
        elif choice == '4':
            print("👋 Exiting system...")
            break
        
        else:
            print("❌ Invalid choice, try again.")
if __name__ == "__main__":
    main()
