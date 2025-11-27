import pandas as pd
from typing import Dict, List
import numpy as np

# ASSIGNMENT: Raeeka
# GOAL: Implement evaluation metrics to prove the models work.
# 
# TASKS:
# 1. Implement `train_test_split_interactions` (time-based or random).
# 2. Implement `compute_rmse` for accuracy.
# 3. Implement `precision_at_k` for ranking quality.
# 4. Run `evaluate_models` to compare Content-Based vs. Collaborative vs. Hybrid.

from src.recommender import ContentRecommender, CollaborativeRecommender
from src.data_loader import (
    load_user_top_tracks,
    load_user_top_artists,
    load_user_top_albums,
)

def train_test_split_interactions(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the interaction data into train and test sets.
    
    TODO (Raeeka): 
    - If possible, use a time-based split (train on past, test on future).
    - Otherwise, use a random split.
    """
    # Placeholder: returning original df as both train and test
    print("TODO: Implement proper train/test split")
    return df, df

def compute_rmse(y_true, y_pred) -> float:
    """
    Calculates Root Mean Squared Error.
    
    TODO (Raeeka): Implement RMSE formula.
    """
    # Placeholder
    return 0.0

def precision_at_k(recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    Calculates Precision@K: Proportion of recommended items that are relevant.
    
    TODO (Raeeka): 
    - Count how many items in `recommended`[:k] are present in `ground_truth`.
    - Return count / k.
    """
    # Placeholder
    return 0.0

def evaluate_models() -> Dict[str, float]:
    """
    Main evaluation loop.
    
    TODO (Raeeka):
    1. Load data using src.data_loader.
    2. Split data into train/test.
    3. Train ContentRecommender and CollaborativeRecommender on Train set.
    4. Generate predictions for users in Test set.
    5. Compute metrics (RMSE, Precision) for each model.
    """
    results = {
        "Content-Based RMSE": 0.0,
        "Collaborative RMSE": 0.0,
        "Hybrid RMSE": 0.0, # If available
        "Content-Based Precision@10": 0.0,
        "Collaborative Precision@10": 0.0
    }
    
    print("Starting evaluation...")
    # ... Logic here ...
    
    return results

if __name__ == "__main__":
    metrics = evaluate_models()
    print("Evaluation Results:", metrics)
