import pytest
import pandas as pd
import numpy as np
import os
from src.data_loader import DataLoader

@pytest.fixture
def sample_data_path(tmp_path):
    # Create a dummy CSV file representing Last.fm data
    # Columns: user_id, artist_id, artist_name, play_count
    df = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user1', 'user3', np.nan],
        'artist_id': ['art1', 'art2', 'art2', 'art3', 'art1'],
        'artist_name': ['Radiohead', 'Coldplay', 'Coldplay', 'Daft Punk', 'Radiohead'],
        'play_count': [10, 50, 20, 100, 5]
    })
    
    file_path = tmp_path / "sample_interactions.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_load_data(sample_data_path):
    """Test if data loads correctly."""
    loader = DataLoader(sample_data_path)
    df = loader.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert 'play_count' in df.columns

def test_clean_data(sample_data_path):
    """Test if missing values are handled (dropped)."""
    loader = DataLoader(sample_data_path)
    loader.load_data()
    cleaned_df = loader.clean_data()
    
    # Should drop the row with NaN user_id
    assert len(cleaned_df) == 4
    assert cleaned_df['user_id'].isnull().sum() == 0

def test_normalize_data(sample_data_path):
    """Test if play_count is normalized to 0-1 range."""
    loader = DataLoader(sample_data_path)
    loader.load_data()
    loader.clean_data()
    normalized_df = loader.normalize_data(method='minmax')
    
    # Check range
    assert normalized_df['norm_play_count'].min() == 0.0
    assert normalized_df['norm_play_count'].max() == 1.0
    
    # Verify relative order (100 plays should be 1.0, 10 plays should be 0.0 in this subset)
    # Note: Min in cleaned data is 10, Max is 100.
    # 10 -> 0.0
    # 100 -> 1.0
    assert normalized_df.loc[normalized_df['play_count'] == 100, 'norm_play_count'].values[0] == 1.0
