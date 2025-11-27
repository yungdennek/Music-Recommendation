import streamlit as st
import pandas as pd

# ASSIGNMENT: Ananya
# GOAL: Create a simple web interface to demo the recommendation system.
#
# TASKS:
# 1. Load the data and initialize the models (Content, Collaborative, Hybrid).
# 2. Create a sidebar for User selection or Song input.
# 3. Display recommended songs/artists in a nice list or table.

# NOTE: Run this app with: `streamlit run app.py`

from src.data_loader import load_lyrics
from src.recommender import ContentRecommender, CollaborativeRecommender
# from src.hybrid_recommender import HybridRecommender # Uncomment when Daniel is done

@st.cache_resource
def load_system():
    """
    Load data and initialize models once.
    TODO (Ananya): Connect this to the real data loader and recommender classes.
    """
    st.write("Loading data...")
    # df = load_lyrics()
    # content_rec = ContentRecommender(df, ...)
    # return content_rec
    return None

def main():
    st.title("🎵 Music Recommendation System")
    st.markdown("CMPE 257 Group Project")

    # Initialize models
    # models = load_system()

    # Sidebar
    st.sidebar.header("User Controls")
    mode = st.sidebar.radio("Recommendation Mode", ["Content-Based (By Song)", "Collaborative (By User)", "Hybrid"])

    if mode == "Content-Based (By Song)":
        st.header("Find Similar Songs")
        song_name = st.text_input("Enter a song name:", "Imagine")
        
        if st.button("Recommend"):
            st.write(f"TODO: Display songs similar to '{song_name}'")
            # results = model.recommend_songs(song_name)
            # st.table(results)

    elif mode == "Collaborative (By User)":
        st.header("User Personalization")
        user_id = st.number_input("Enter User ID:", min_value=1, max_value=1000, value=1)
        
        if st.button("Get Recommendations"):
            st.write(f"TODO: Display recommendations for User {user_id}")
            # results = collab_model.get_top_items(user_id)
            # st.write(results)

    elif mode == "Hybrid":
        st.header("Hybrid Recommendations")
        st.info("Combines both approaches for better accuracy.")
        # TODO: Hook up Daniel's hybrid recommender here

if __name__ == "__main__":
    main()
