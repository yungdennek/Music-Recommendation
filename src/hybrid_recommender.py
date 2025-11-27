from typing import List, Dict
import numpy as np

# ASSIGNMENT: Daniel
# GOAL: Combine Content-Based and Collaborative Filtering into a Hybrid model.
#
# TASKS:
# 1. Implement `HybridRecommender` class.
# 2. Implement `recommend_for_user`: Weighted average of scores or rank aggregation.
# 3. Implement `handle_cold_start`: Fallback logic for new users.

from src.recommender import ContentRecommender, CollaborativeRecommender

class HybridRecommender:
    def __init__(
        self,
        content_rec: ContentRecommender,
        collab_rec: CollaborativeRecommender,
        alpha: float = 0.5,
    ):
        """
        Args:
            content_rec: Initialized ContentRecommender instance.
            collab_rec: Initialized CollaborativeRecommender instance.
            alpha: Weight for Collaborative Filtering (0.0 to 1.0).
                   Final Score = alpha * Collab_Score + (1 - alpha) * Content_Score
        """
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.alpha = alpha

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> List[str]:
        """
        Generates hybrid recommendations.
        
        TODO (Daniel):
        1. Get candidates/scores from self.collab_rec.
        2. Get candidates/scores from self.content_rec (e.g., based on user's top tracks).
        3. Normalize scores if they are on different scales!
        4. Combine scores: score = alpha * cf_score + (1-alpha) * cb_score.
        5. Return top_n items.
        """
        # Placeholder logic
        print(f"TODO: Generate hybrid recommendations for User {user_id}")
        return []

    def handle_cold_start(self, user_id: int, top_n: int = 10) -> List[str]:
        """
        Handles cases where user has no/little history.
        
        TODO (Daniel):
        1. Check if user exists or has enough interactions.
        2. If not, return popular items or generic content-based recommendations.
        """
        print(f"TODO: Handle cold start for User {user_id}")
        return []

if __name__ == "__main__":
    # Test stub
    print("Hybrid Recommender initialized.")
