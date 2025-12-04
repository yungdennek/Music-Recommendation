# Presentation Roadmap

## Overview
- **Duration**: 15 minutes total
- **Format**: Slides + Live Demo
- **Deadline**: Thursday, Dec 4, 2025

---

## Slide Structure

| # | Slide Title | Time | Owner | Content |
|---|-------------|------|-------|---------|
| 1 | Title | 15s | All | Project name, team members, CMPE 257 |
| 2 | Problem & Motivation | 2 min | Eugene | Choice paralysis, engagement, discovery |
| 3 | Datasets | 1 min | Ananya | Yambda + Last.fm 360K overview |
| 4 | Preprocessing | 1 min | Ananya | Filtering, normalization, TF-IDF |
| 5 | Content-Based Filtering | 1.5 min | Eugene | Metadata + cosine similarity |
| 6 | Collaborative Filtering | 1.5 min | George | User-User similarity, TF-IDF profiles |
| 7 | Hybrid Approach | 1.5 min | George/Daniel | Weighted combination, cold-start handling |
| 8 | Evaluation Methodology | 1 min | Raeeka | 80/20 split, Precision@K, RMSE |
| 9 | Results | 2 min | Raeeka | Bar charts comparing models |
| 10 | Demo | 3 min | Ananya | Live Streamlit walkthrough |
| 11 | Challenges & Future Work | 1 min | All | Cold start, scalability, deep learning |

---

## Slide Content Details

### Slide 1: Title
- Music Recommendation System
- Team: Eugene Lacatis, Scott Kennedy, Ananya Makwana, George Luu, Raeeka Yusuf
- CMPE 257 - Machine Learning
- San Jose State University

### Slide 2: Problem & Motivation
**Key Points:**
- Modern streaming platforms have 100M+ tracks
- Users suffer from choice paralysis
- Simple popularity charts miss individual taste
- Goal: Predict user preferences + enable discovery

**Business Case:**
- Increased engagement and session time
- Higher conversion for premium features
- Better ad exposure through longer platform usage

### Slide 3: Datasets
| Dataset | Size | Features |
|---------|------|----------|
| Yambda | ~4.79B interactions, ~1M users, ~9.39M tracks | user_id, item_id, timestamp, event_type, played_ratio |
| Last.fm 360K | ~360K users, ~17M events, ~160K artists | user_id, artist_id, artist_name, play_count |

**Note:** Using manageable subsets for experimentation

### Slide 4: Preprocessing
- Filter and aggregate multi-event logs (listens, likes, dislikes)
- Normalize interaction strength (played ratio, track length)
- Build user-item interaction matrices
- Construct TF-IDF vectors from text fields (tags, artist names)
- Clean and normalize artist/track names

### Slide 5: Content-Based Filtering
**Approach:**
- Build item profiles from metadata (artist, genre, tags)
- Use TF-IDF vectorization for text features
- Calculate cosine similarity between items
- Recommend items most similar to user's history

**Formula:**
```
similarity(A, B) = (A . B) / (||A|| * ||B||)
```

**Visual:** [PLACEHOLDER: Similarity matrix heatmap or diagram]

### Slide 6: Collaborative Filtering
**Approach:**
- Find users with similar listening patterns
- User-User similarity based on top tracks/artists/albums
- TF-IDF weighted comparison of user profiles
- Recommend items that similar users enjoyed

**Visual:** [PLACEHOLDER: User similarity diagram]

### Slide 7: Hybrid Approach
**Strategy:**
```
Final_Score = alpha * CF_Score + (1 - alpha) * CB_Score
```
- Default alpha = 0.5 (tunable)

**Cold Start Handling:**
- New users (< 5 interactions): Fall back to content-based + popularity
- New items: Use content features only

**Visual:** [PLACEHOLDER: Hybrid architecture diagram]

### Slide 8: Evaluation Methodology
**Protocol:**
- 80/20 train/test split (random or time-based)
- Hold out portion of user interactions
- Attempt to recover hidden items via recommendations

**Metrics:**
- Precision@K: Proportion of recommended items that are relevant
- Recall@K: Proportion of relevant items that are recommended
- RMSE/MAE: Prediction accuracy (if explicit ratings used)

### Slide 9: Results
[PLACEHOLDER: Awaiting Raeeka's evaluation results]

**Expected Visualizations:**
1. Bar chart: Precision@10 for CB vs CF vs Hybrid
2. Bar chart: RMSE comparison across models
3. Table: Summary metrics

| Model | Precision@10 | RMSE |
|-------|--------------|------|
| Content-Based | [TBD] | [TBD] |
| Collaborative | [TBD] | [TBD] |
| Hybrid | [TBD] | [TBD] |
| Popularity Baseline | [TBD] | [TBD] |

### Slide 10: Demo
**Flow:**
1. User enters a song or artist name
2. System displays recommended tracks
3. Show "Cold Start" example (new user)
4. Show "Active User" example (user with history)

**Tool:** Streamlit app (`app.py`)

### Slide 11: Challenges & Future Work
**Challenges Encountered:**
- Data sparsity in user-item matrix
- Cold start for new users/items
- Scalability of similarity computations
- Popularity bias in recommendations

**Future Directions:**
- Deep learning (RNNs for session-based recommendations)
- Audio feature embeddings (spectral analysis)
- Real-time updates and feedback loops
- A/B testing framework

---

## Required Visuals Checklist

- [ ] Data distribution chart (user interaction counts)
- [ ] Track popularity distribution
- [ ] Content-Based similarity diagram or formula
- [ ] Collaborative Filtering user similarity diagram
- [ ] Hybrid architecture diagram
- [ ] Results bar charts (Precision@K, RMSE)
- [ ] Demo screenshots or live walkthrough

---

## Presentation Tips

1. **Practice transitions** between speakers
2. **Have backup screenshots** of demo in case of technical issues
3. **Keep slides visual** - minimize text, maximize diagrams/charts
4. **Prepare for Q&A** on: Why these datasets? Why this hybrid weight? How does cold start work?
