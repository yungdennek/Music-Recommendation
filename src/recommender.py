import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict

class ContentRecommender:
    def __init__(self, df: pd.DataFrame, similarity_matrix):
        self.df = df
        self.similarity_matrix = similarity_matrix

    def recommend_songs(self, song_name: str, top_n: int = 4) -> List[str]:
        try:
            idx = self.df.loc[self.df["song"] == song_name].index[0]
        except IndexError:
            return []
            
        distances = sorted(list(enumerate(self.similarity_matrix[idx])), reverse=True, key=lambda x: x[1])
        
        songs = []
        for i in distances[1:top_n+1]:
            songs.append(self.df.iloc[i[0]].song)
        return songs

    def recommend_artists(self, artist_name: str, top_n: int = 4) -> List[str]:
        try:
            idx = self.df.loc[self.df["artist"] == artist_name].index[0]
        except IndexError:
            return []
            
        distances = sorted(list(enumerate(self.similarity_matrix[idx])), reverse=True, key=lambda x: x[1])
        
        artists = []
        for i in distances[1:top_n+1]:
            artists.append(self.df.iloc[i[0]].artist)
        return artists

class CollaborativeRecommender:
    def __init__(self, tracks_df: pd.DataFrame, artists_df: pd.DataFrame, albums_df: pd.DataFrame):
        self.tracks_df = tracks_df
        self.artists_df = artists_df
        self.albums_df = albums_df

    def get_top_items(self, user_id: int, category: str) -> List[str]:
        if category == 'songs':
            return self.tracks_df.loc[self.tracks_df["user_id"] == user_id, "track_name"].tolist()
        elif category == 'artists':
            return self.artists_df.loc[self.artists_df["user_id"] == user_id, "artist_name"].tolist()
        elif category == 'albums':
            return self.albums_df.loc[self.albums_df["user_id"] == user_id, "album_name"].tolist()
        return []

    def calculate_user_similarity(self, user1: int, user2: int) -> int:
        u1_songs = self.get_top_items(user1, 'songs')
        u2_songs = self.get_top_items(user2, 'songs')
        
        u1_artists = self.get_top_items(user1, 'artists')
        u2_artists = self.get_top_items(user2, 'artists')
        
        u1_albums = self.get_top_items(user1, 'albums')
        u2_albums = self.get_top_items(user2, 'albums')

        score = 0
        for i in range(min(50, len(u1_songs))):
            if u1_songs[i] in u2_songs:
                score += 1
            if i < len(u1_artists) and u1_artists[i] in u2_artists:
                score += 1
            if i < len(u1_albums) and u1_albums[i] in u2_albums:
                score += 1
        return score

    def calculate_tfidf_user_similarity(self, user1: int, user2: int) -> float:
        """
        Calculates similarity between two users based on TF-IDF vectors of their top items.
        Returns the sum of cosine similarities for songs, artists, and albums.
        """
        songs1 = self.get_top_items(user1, 'songs')
        songs2 = self.get_top_items(user2, 'songs')
        artists1 = self.get_top_items(user1, 'artists')
        artists2 = self.get_top_items(user2, 'artists')
        albums1 = self.get_top_items(user1, 'albums')
        albums2 = self.get_top_items(user2, 'albums')

        # Handle empty lists cases to avoid empty vocabulary errors
        if not (songs1 + songs2) or not (artists1 + artists2) or not (albums1 + albums2):
            return 0.0

        vectorizer = TfidfVectorizer()
        
        try:
            tfidf_matrix_songs = vectorizer.fit_transform(songs1 + songs2)
            tfidf_matrix_artists = vectorizer.fit_transform(artists1 + artists2)
            tfidf_matrix_albums = vectorizer.fit_transform(albums1 + albums2)
        except ValueError:
            # Handles cases where vocab might be empty
            return 0.0

        # Separate the matrices for each list
        tfidf_list_songs1 = tfidf_matrix_songs[:len(songs1)]
        tfidf_list_songs2 = tfidf_matrix_songs[len(songs1):]
        tfidf_list_artists1 = tfidf_matrix_artists[:len(artists1)]
        tfidf_list_artists2 = tfidf_matrix_artists[len(artists1):]
        tfidf_list_albums1 = tfidf_matrix_albums[:len(albums1)]
        tfidf_list_albums2 = tfidf_matrix_albums[len(albums1):]

        # Calculate cosine similarity
        sim_songs = np.sum(cosine_similarity(tfidf_list_songs1, tfidf_list_songs2))
        sim_artists = np.sum(cosine_similarity(tfidf_list_artists1, tfidf_list_artists2))
        sim_albums = np.sum(cosine_similarity(tfidf_list_albums1, tfidf_list_albums2))
        
        return sim_songs + sim_artists + sim_albums

    def get_similar_users_tfidf(self, user_id: int, threshold: float = 0.1, limit: int = 100) -> List[int]:
        """Finds users similar to the given user using the TF-IDF metric."""
        similar_users = []
        
        # Calculate self-similarity for normalization
        self_sim = self.calculate_tfidf_user_similarity(user_id, user_id)
        if self_sim == 0:
            return []

        # Search through first 'limit' users (optimization for demo purposes)
        # In production, you'd pre-calculate this or use a nearest neighbor index
        for i in range(1, limit + 1):
            if i == user_id:
                continue
                
            sim = self.calculate_tfidf_user_similarity(user_id, i)
            ratio = sim / self_sim
            
            if ratio > threshold:
                similar_users.append(i)
                
        return similar_users

    def get_shared_items(self, user1: int, user2: int, category: str) -> List[str]:
        items1 = self.get_top_items(user1, category)
        items2 = self.get_top_items(user2, category)
        return [item for item in items1 if item in items2]

    def recommend_new_music(self, user_id: int, threshold: float) -> Dict[str, List[str]]:
        # Note: This implements the logic from user_coefficients/new_music in the notebook
        # Ideally this should be optimized, as scanning user IDs 1-100 is arbitrary
        similar_users = []
        # Simplified similarity check based on notebook logic
        for i in range(1, 100):
            if i == user_id: continue
            # Using a simplified metric here as the notebook's full TF-IDF user sim is heavy
            # Fallback to basic similarity for now or use the one defined
            sim_score = self.calculate_user_similarity(user_id, i)
            if sim_score > threshold:
                similar_users.append(i)
        
        recommendations = {
            'songs': [],
            'artists': [],
            'albums': []
        }
        
        current_songs = set(self.get_top_items(user_id, 'songs'))
        current_artists = set(self.get_top_items(user_id, 'artists'))
        current_albums = set(self.get_top_items(user_id, 'albums'))

        for sim_user in similar_users:
            # Add items from similar user that current user doesn't have
            new_songs = [s for s in self.get_top_items(sim_user, 'songs') if s not in current_songs]
            new_artists = [a for a in self.get_top_items(sim_user, 'artists') if a not in current_artists]
            new_albums = [a for a in self.get_top_items(sim_user, 'albums') if a not in current_albums]
            
            recommendations['songs'].extend(new_songs)
            recommendations['artists'].extend(new_artists)
            recommendations['albums'].extend(new_albums)

        # Deduplicate
        return {k: list(set(v)) for k, v in recommendations.items()}
