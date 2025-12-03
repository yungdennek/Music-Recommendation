# Implementation Roadmap

## Status Overview

| Component | Status | Owner | File |
|-----------|--------|-------|------|
| Data Acquisition | Done | Ananya | `data_loader.py` |
| Content-Based Filtering | Done | Eugene | `recommender.py` |
| Collaborative Filtering | Mostly Done | George | `recommender.py` |
| Hybrid Recommender | In Progress | George/Daniel | `hybrid_recommender.py` |
| Evaluation Framework | In Progress | Raeeka | `evaluation.py` |
| Matrix Factorization | Not Started | Scott | TBD |
| Demo App | Exists | Ananya | `app.py` |

**Deadline**: Thursday, Dec 4, 2025

---

## Critical Path

### P0 - Must Complete (Blocking)

#### 1. Evaluation Framework (`evaluation.py`) - Owner: Raeeka
- [ ] Implement `train_test_split_interactions()` - time-based or random split
- [ ] Implement `compute_rmse()` - RMSE formula
- [ ] Implement `precision_at_k()` - count relevant items in top K
- [ ] Implement `evaluate_models()` - main loop to generate results
- [ ] Output results to console/file for report

#### 2. Hybrid Recommender (`hybrid_recommender.py`) - Owner: George
- [ ] Implement `recommend_for_user()` - weighted combination of CF + CB scores
- [ ] Implement `handle_cold_start()` - fallback logic for new users
- [ ] Normalize scores from different models before combining
- [ ] Test with sample users

### P1 - Should Complete

#### 3. Score Exposure for Hybrid Integration - Owner: Eugene
- [ ] Add `get_similarity_scores()` method to `ContentRecommender`
- [ ] Returns `List[Tuple[str, float]]` instead of just names
- [ ] Enables weighted combination in hybrid model

#### 4. Demo App Verification - Owner: Ananya
- [ ] Verify `app.py` runs with current recommenders
- [ ] Test content-based recommendations
- [ ] Test collaborative recommendations
- [ ] Add hybrid recommendations if available

### P2 - Nice to Have

#### 5. Matrix Factorization - Owner: Scott
- [ ] Implement SVD or ALS using Surprise library
- [ ] Integrate into hybrid model as third signal
- [ ] Compare performance against CF/CB

---

## File-by-File Implementation Details

### `src/evaluation.py`

```python
# Current state: Skeleton with TODOs
# Needs implementation of:

def train_test_split_interactions(df, test_ratio=0.2):
    # Option A: Random split
    # Option B: Time-based split (preferred for recsys)
    pass

def compute_rmse(y_true, y_pred):
    # RMSE = sqrt(mean((y_true - y_pred)^2))
    pass

def precision_at_k(recommended, ground_truth, k=10):
    # hits = len(set(recommended[:k]) & set(ground_truth))
    # return hits / k
    pass

def evaluate_models():
    # 1. Load data
    # 2. Split train/test
    # 3. Train models on train set
    # 4. Generate predictions on test set
    # 5. Compute and return metrics
    pass
```

### `src/hybrid_recommender.py`

```python
# Current state: Skeleton with TODOs
# Needs implementation of:

def recommend_for_user(self, user_id, top_n=10):
    # 1. Get CF candidates with scores
    # 2. Get CB candidates with scores (based on user's top tracks)
    # 3. Normalize scores to [0, 1] range
    # 4. Combine: score = alpha * cf + (1-alpha) * cb
    # 5. Sort and return top_n
    pass

def handle_cold_start(self, user_id, top_n=10):
    # 1. Check user interaction count
    # 2. If < threshold: return popularity-based or CB-only
    # 3. Else: use normal hybrid logic
    pass
```

### `src/recommender.py`

```python
# Needs addition for hybrid integration:

def get_similarity_scores(self, song_name, top_n=10):
    # Returns List[Tuple[str, float]] with scores
    # Enables weighted combination in hybrid
    pass
```

---

## Timeline

### Monday, Dec 2 (Today)
- [ ] Raeeka: Start `evaluation.py` implementation
- [ ] George: Start `hybrid_recommender.py` implementation
- [ ] Eugene: Add `get_similarity_scores()`, create visualizations
- [ ] Scott: Attempt SVD integration (optional)

### Tuesday, Dec 3
- [ ] Complete all P0 tasks
- [ ] Run full evaluation, collect numbers
- [ ] Build presentation slides with actual results
- [ ] Fill in report sections V, VI, VII

### Wednesday, Dec 3 (Evening)
- [ ] Final testing of demo app
- [ ] Polish presentation slides
- [ ] Complete project report

### Thursday, Dec 4
- [ ] Rehearse presentation
- [ ] Submit report
- [ ] Present

---

## Integration Points

### How Components Connect

```
data_loader.py
     |
     v
+--------------------+     +------------------------+
| ContentRecommender | --> |                        |
+--------------------+     |   HybridRecommender    |
                           |                        |
+------------------------+ |  (weighted combination)|
| CollaborativeRecommender|-->                      |
+------------------------+ +------------------------+
                                      |
                                      v
                              +-------------+
                              | evaluation.py|
                              +-------------+
                                      |
                                      v
                              +-------------+
                              |   app.py    |
                              +-------------+
```

### Data Flow for Evaluation

1. `data_loader.py` loads user interactions
2. `evaluation.py` splits into train/test
3. Models trained on train set
4. Models generate recommendations for test users
5. Compare recommendations against held-out items
6. Compute Precision@K, RMSE

---

## Testing Commands

```bash
# Run evaluation
python -m src.evaluation

# Run demo app
streamlit run app.py

# Run specific recommender tests
python -c "from src.recommender import ContentRecommender; print('CB OK')"
python -c "from src.recommender import CollaborativeRecommender; print('CF OK')"
```

---

## Contingency Plans

### If Hybrid Not Ready
- Present CB and CF as separate models
- Show comparison without hybrid
- Mention hybrid as "planned integration"

### If Evaluation Numbers Not Ready
- Use placeholder metrics in slides
- Focus demo on qualitative results
- Explain methodology without specific numbers

### If Demo Breaks
- Have screenshots ready
- Pre-record a backup video
- Show notebook outputs instead
