import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    return df

df = load_data()

# Features
features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in features:
    df[feature] = df[feature].fillna('')

# Combine Features
combined_features = (
    df['genres'] + ' ' +
    df['keywords'] + ' ' +
    df['tagline'] + ' ' +
    df['cast'] + ' ' +
    df['director']
)

# TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Similarity Matrix
similarity = cosine_similarity(feature_vectors)

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter Movie Name")

if st.button("Recommend Movies"):

    if movie_name:

        list_of_titles = df['title'].tolist()

        find_close_match = difflib.get_close_matches(
            movie_name,
            list_of_titles
        )

        if len(find_close_match) == 0:
            st.error("Movie not found!")
        else:

            close_match = find_close_match[0]

            index_of_movie = df[df.title == close_match].index[0]

            similarity_score = list(
                enumerate(similarity[index_of_movie])
            )

            sorted_similar_movies = sorted(
                similarity_score,
                key=lambda x: x[1],
                reverse=True
            )

            st.success(f"Top 30 Movies similar to {close_match}")

            count = 1

            for movie in sorted_similar_movies:

                index = movie[0]
                title = df.iloc[index]['title']

                if count <= 30:
                    st.write(f"{count}. {title}")
                    count += 1