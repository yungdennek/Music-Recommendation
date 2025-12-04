import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_lyrics, load_user_top_tracks, load_user_top_artists


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def create_visualizations():
    """Generate all visualizations for the report"""
    
    
    df_tracks = load_user_top_tracks()
    df_artists = load_user_top_artists()
    df_lyrics = load_lyrics()
    
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    print("Generating visualizations...")
    
# Chart 1- User Activity Distribution
    print("Creating Chart 1: User Activity Distribution...")
    user_track_counts = df_tracks.groupby('user_id').size()
    
    plt.figure(figsize=(12, 6))
    plt.hist(user_track_counts, bins=50, color='steelblue', edgecolor='black')
    plt.xlabel('Number of Tracks per User', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.title('Distribution of User Activity (Track Counts)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/user_activity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved user_activity_distribution.png")
    
    # Chart 2-- Top 20 Most Popular Artists
    print("Creating Chart 2: Top Artists...")
    top_artists = df_artists['artist_name'].value_counts().head(20)
    
    plt.figure(figsize=(12, 8))
    top_artists.plot(kind='barh', color='coral')
    plt.xlabel('Number of Users Listening', fontsize=12)
    plt.ylabel('Artist', fontsize=12)
    plt.title('Top 20 Most Popular Artists Across Users', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('visualizations/top_artists.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved top_artists.png")
    
    # Chart 3-Dataset Overview
    print("Creating Chart 3: Dataset Overview...")
    dataset_stats = pd.DataFrame({
        'Dataset': ['Songs (Lyrics)', 'User Tracks', 'User Artists', 'Unique Users'],
        'Count': [len(df_lyrics), len(df_tracks), len(df_artists), df_tracks['user_id'].nunique()]
    })
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(dataset_stats['Dataset'], dataset_stats['Count'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('Count', fontsize=12)
    plt.title('Dataset Overview: Data Volume', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved dataset_overview.png")
    
    print("\nAll visualizations saved in 'visualizations/' folder!")

if __name__ == "__main__":
    create_visualizations()