{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "029702f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-03-16 14:38:46.295 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\User1\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-03-16 14:38:51.435 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "st.title(\"🎬 Movie Recommender System\")\n",
    "\n",
    "# Load datasets\n",
    "movies = pd.read_csv(\"movies.csv\")\n",
    "ratings = pd.read_csv(\"ratings.csv\")\n",
    "\n",
    "# Merge datasets\n",
    "data = pd.merge(ratings, movies, on=\"movieId\")\n",
    "\n",
    "# Create user-movie matrix\n",
    "user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')\n",
    "\n",
    "# Fill missing values\n",
    "user_movie_matrix = user_movie_matrix.fillna(0)\n",
    "\n",
    "# Compute similarity between movies\n",
    "movie_similarity = cosine_similarity(user_movie_matrix.T)\n",
    "movie_similarity_df = pd.DataFrame(movie_similarity,\n",
    "                                   index=user_movie_matrix.columns,\n",
    "                                   columns=user_movie_matrix.columns)\n",
    "\n",
    "# Movie selection\n",
    "movie_list = movies['title'].values\n",
    "selected_movie = st.selectbox(\"Select a movie you like:\", movie_list)\n",
    "\n",
    "# Recommendation function\n",
    "def recommend_movies(movie_name, num_recommendations=5):\n",
    "    similar_scores = movie_similarity_df[movie_name].sort_values(ascending=False)\n",
    "    recommendations = similar_scores.iloc[1:num_recommendations+1]\n",
    "    return recommendations.index\n",
    "\n",
    "# Show recommendations\n",
    "if st.button(\"Recommend Movies\"):\n",
    "    recommendations = recommend_movies(selected_movie)\n",
    "\n",
    "    st.subheader(\"Recommended Movies For You\")\n",
    "\n",
    "    for movie in recommendations:\n",
    "        st.write(movie)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e753685e",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (3737097518.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"C:\\Users\\User1\\AppData\\Local\\Temp\\ipykernel_30112\\3737097518.py\"\u001b[1;36m, line \u001b[1;32m1\u001b[0m\n\u001b[1;33m    streamlit run app.py\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "streamlit run app.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7638484a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
