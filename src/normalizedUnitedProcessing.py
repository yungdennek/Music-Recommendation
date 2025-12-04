import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def standardize_text(text): # remove extra whitespace and convert to lowercase
    if pd.isna(text):
        return text
    return ' '.join(str(text).lower().strip().split())

def preprocess_user_top_tracks(filepath):
    df = pd.read_csv(filepath)
    df = df.drop('mbid', axis=1, errors='ignore')
    df = df.dropna()
    df = df[df['rank'] <= 30]
    df['artist_name'] = df['artist_name'].apply(standardize_text)
    df['track_name'] = df['track_name'].apply(standardize_text)
    
    return df

def preprocess_user_top_artists(filepath):
    df = pd.read_csv(filepath)
    df = df.drop('mbid', axis=1, errors='ignore')
    df = df.dropna()
    df = df[df['rank'] <= 30]
    df['artist_name'] = df['artist_name'].apply(standardize_text)
    
    return df

def preprocess_user_top_albums(filepath):
    df = pd.read_csv(filepath)
    df = df.drop('mbid', axis=1, errors='ignore')
    df = df.dropna()
    df = df[df['rank'] <= 30]
    df['artist_name'] = df['artist_name'].apply(standardize_text)
    df['album_name'] = df['album_name'].apply(standardize_text)

    return df

def preprocess_spotify_millsongdata(filepath):
    """Preprocess Spotify million song dataset"""
    df = pd.read_csv(filepath)
    df = df.drop('link', axis=1, errors='ignore')
    df = df.dropna()
    df['artist'] = df['artist'].apply(standardize_text)
    df['song'] = df['song'].apply(standardize_text)
    
    return df
######################################
#united dataset 
def remove_low_playcount(df, min_playcount=2):
    print(f"\nRemoving entries with playcount < {min_playcount}.")
    original_len = len(df)
    df = df[df['playcount'] >= min_playcount]
    return df

def filter_sparse_users(df, min_interactions=5):
    """Remove users with fewer than min_interactions"""
    print(f"\nFiltering users with < {min_interactions} interactions.")
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index

    df = df[df['user_id'].isin(valid_users)]
    
    
    return df

def filter_sparse_tracks(df, min_users=3):
    print(f"\nFiltering tracks with < {min_users} listeners.")
    track_counts = df.groupby(['track_name', 'artist_name']).size()
    valid_tracks = track_counts[track_counts >= min_users].index
    
    original_len = len(df)
    original_tracks = len(df.groupby(['track_name', 'artist_name']))
    
    df['track_artist_combo'] = list(zip(df['track_name'], df['artist_name']))
    df = df[df['track_artist_combo'].isin(valid_tracks)]
    df = df.drop('track_artist_combo', axis=1)
    
    return df

def normalize_playcounts(df):
    """Normalize playcounts per user using min-max scaling (FAST version)"""
    print("\nNormalizing playcounts per user.")
    
    # Store original playcount for reference
    df['original_playcount'] = df['playcount']
    
    # Fast vectorized normalization per user
    df['normalized_playcount'] = df.groupby('user_id')['playcount'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )
    
    return df

def create_implicit_feedback(df, threshold=0.2):
    """Create binary implicit feedback (1 = user engaged with track, 0 = not)"""
    print(f"\nCreating implicit feedback (threshold={threshold})...")
    df['implicit_feedback'] = (df['normalized_playcount'] >= threshold).astype(int)
    positive_interactions = df['implicit_feedback'].sum()
    total_interactions = len(df)
    
    print(f"  Positive interactions: {positive_interactions} ({positive_interactions/total_interactions*100:.1f}%)")
    print(f"  Negative interactions: {total_interactions - positive_interactions} ({(total_interactions - positive_interactions)/total_interactions*100:.1f}%)")
    
   
    return df
"""
def merge_with_spotify_data(df, spotify_df):
    print("\nMerging with Spotify data to add lyrics...")
    
    # Merge on artist and track name
    df = df.merge(
        spotify_df[['artist', 'song', 'text']],
        left_on=['artist_name', 'track_name'],
        right_on=['artist', 'song'],
        how='left'
    )
    
    # Drop duplicate columns
    df = df.drop(['artist', 'song'], axis=1, errors='ignore')
    
    # Rename text column to lyrics
    df = df.rename(columns={'text': 'lyrics'})
    
    tracks_with_lyrics = df['lyrics'].notna().sum()
    print(f"  Tracks with lyrics: {tracks_with_lyrics} ({tracks_with_lyrics/len(df)*100:.1f}%)")
    
    return df
"""
def main(skip_preprocessing=True):
    """Complete data processing pipeline"""
    
    if skip_preprocessing:
        print("123 skipping preprocessing")    
        tracks_df = pd.read_csv('preprocessed_user_top_tracks.csv')
        artists_df = pd.read_csv('preprocessed_user_top_artists.csv')
        albums_df = pd.read_csv('preprocessed_user_top_albums.csv')
        spotify_df = pd.read_csv('preprocessed_spotify_millsongdata.csv')
        
    else:
        print("123 preprocessing")
        tracks_df = preprocess_user_top_tracks('user_top_tracks.csv')
        artists_df = preprocess_user_top_artists('user_top_artists.csv')
        albums_df = preprocess_user_top_albums('user_top_albums.csv')
        spotify_df = preprocess_spotify_millsongdata('spotify_millsongdata.csv')
        
        # Save preprocessed files
        print("\nSaving preprocessed files.")
        tracks_df.to_csv('preprocessed_user_top_tracks.csv', index=False)
        artists_df.to_csv('preprocessed_user_top_artists.csv', index=False)
        albums_df.to_csv('preprocessed_user_top_albums.csv', index=False)
        spotify_df.to_csv('preprocessed_spotify_millsongdata.csv', index=False)
        print("Preprocessed files saved")
    print("uniting dataset")
    df = tracks_df.copy()
    
    df = remove_low_playcount(df, min_playcount=2)
    df = filter_sparse_users(df, min_interactions=5)
    df = filter_sparse_tracks(df, min_users=3)
    df = normalize_playcounts(df)
    df = create_implicit_feedback(df, threshold=0.15)
    #df = merge_with_spotify_data(df, spotify_df)
    
    column_order = [
        'user_id', 'rank', 'track_name', 'artist_name', 
        'original_playcount', 'normalized_playcount', 'implicit_feedback',
    ]
    df = df[column_order]
    
    # Save unified dataset
    print("Saving unified dataset.")
    df.to_csv('unified_dataset.csv', index=False)
    


if __name__ == "__main__":
    import sys
    
    skip = '--skip-preprocessing' in sys.argv or '-s' in sys.argv
    
    if skip:
        print("Running with existing preprocessed files\n")
    
    main(skip_preprocessing=skip)