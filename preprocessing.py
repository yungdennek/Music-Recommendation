import pandas as pd
import numpy as np

def standardize_text(text):
    """Standardize text by converting to lowercase and removing extra whitespace"""
    if pd.isna(text):
        return text
    return ' '.join(str(text).lower().strip().split())

def preprocess_user_top_tracks(filepath):
    """Preprocess user top tracks dataset"""
    print(f"Processing {filepath}...")
    df = pd.read_csv(filepath)
    
    # Remove mbid column
    df = df.drop('mbid', axis=1, errors='ignore')
    
    # Remove null values
    df = df.dropna()
    
    # Keep only entries with rank <= 30
    df = df[df['rank'] <= 30]
    
    # Standardize artist and track names
    df['artist_name'] = df['artist_name'].apply(standardize_text)
    df['track_name'] = df['track_name'].apply(standardize_text)
    
    print(f"  Original rows: {len(pd.read_csv(filepath))}, After preprocessing: {len(df)}")
    return df

def preprocess_user_top_artists(filepath):
    """Preprocess user top artists dataset"""
    print(f"Processing {filepath}...")
    df = pd.read_csv(filepath)
    
    # Remove mbid column
    df = df.drop('mbid', axis=1, errors='ignore')
    
    # Remove null values
    df = df.dropna()
    
    # Keep only entries with rank <= 30
    df = df[df['rank'] <= 30]
    
    # Standardize artist names
    df['artist_name'] = df['artist_name'].apply(standardize_text)
    
    print(f"  Original rows: {len(pd.read_csv(filepath))}, After preprocessing: {len(df)}")
    return df

def preprocess_user_top_albums(filepath):
    """Preprocess user top albums dataset"""
    print(f"Processing {filepath}...")
    df = pd.read_csv(filepath)
    
    # Remove mbid column
    df = df.drop('mbid', axis=1, errors='ignore')
    
    # Remove null values
    df = df.dropna()
    
    # Keep only entries with rank <= 30
    df = df[df['rank'] <= 30]
    
    # Standardize artist and album names
    df['artist_name'] = df['artist_name'].apply(standardize_text)
    df['album_name'] = df['album_name'].apply(standardize_text)
    
    print(f"  Original rows: {len(pd.read_csv(filepath))}, After preprocessing: {len(df)}")
    return df

def preprocess_spotify_millsongdata(filepath):
    """Preprocess Spotify million song dataset"""
    print(f"Processing {filepath}...")
    df = pd.read_csv(filepath)
    
    # Remove link column
    df = df.drop('link', axis=1, errors='ignore')
    
    # Remove null values
    df = df.dropna()
    
    # Standardize artist and song names
    df['artist'] = df['artist'].apply(standardize_text)
    df['song'] = df['song'].apply(standardize_text)
    
    print(f"  Original rows: {len(pd.read_csv(filepath))}, After preprocessing: {len(df)}")
    return df

def main():
    """Main preprocessing pipeline"""
    print("Starting data preprocessing...\n")
    
    # Process each file
    tracks_df = preprocess_user_top_tracks('user_top_tracks.csv')
    artists_df = preprocess_user_top_artists('user_top_artists.csv')
    albums_df = preprocess_user_top_albums('user_top_albums.csv')
    spotify_df = preprocess_spotify_millsongdata('spotify_millsongdata.csv')
    
    # Save preprocessed files
    print("\nSaving preprocessed files...")
    tracks_df.to_csv('preprocessed_user_top_tracks.csv', index=False)
    artists_df.to_csv('preprocessed_user_top_artists.csv', index=False)
    albums_df.to_csv('preprocessed_user_top_albums.csv', index=False)
    spotify_df.to_csv('preprocessed_spotify_millsongdata.csv', index=False)
    
    print("\nPreprocessing complete!")
    print("\nOutput files:")
    print("  - preprocessed_user_top_tracks.csv")
    print("  - preprocessed_user_top_artists.csv")
    print("  - preprocessed_user_top_albums.csv")
    print("  - preprocessed_spotify_millsongdata.csv")
    
    # Display summary statistics
    print("\n=== Summary Statistics ===")
    print(f"User Top Tracks: {len(tracks_df)} rows")
    print(f"User Top Artists: {len(artists_df)} rows")
    print(f"User Top Albums: {len(albums_df)} rows")
    print(f"Spotify Million Songs: {len(spotify_df)} rows")

if __name__ == "__main__":
    main()