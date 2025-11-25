import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads data from CSV."""
        # In a real scenario, we might need to handle different delimiters (tab vs comma)
        # Last.fm dataset is often tab-separated, but for now we stick to the test case (CSV)
        try:
            self.df = pd.read_csv(self.file_path)
            return self.df
        except Exception as e:
            raise IOError(f"Error loading file: {e}")

    def clean_data(self):
        """Drops rows with missing values."""
        if self.df is not None:
            self.df = self.df.dropna()
            return self.df
        return None

    def normalize_data(self, method='minmax'):
        """Normalizes the play_count column."""
        if self.df is None:
            return None

        if 'play_count' not in self.df.columns:
            raise ValueError("Column 'play_count' not found in data")

        if method == 'minmax':
            # Manual MinMax Scaling to avoid sklearn dependency
            col = self.df['play_count'].astype(float)
            min_val = col.min()
            max_val = col.max()
            
            if max_val - min_val == 0:
                self.df['norm_play_count'] = 0.0
            else:
                self.df['norm_play_count'] = (col - min_val) / (max_val - min_val)
        
        return self.df
