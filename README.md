# Book Recommendation System

This project implements a Book Recommendation System using multiple recommendation techniques to provide personalized book suggestions.

The project currently includes:

* Content-Based Recommendation using TF-IDF and cosine similarity
* Collaborative Filtering using Surprise library (SVD & KNNBasic)
* Hybrid Recommendation System (planned)

The goal is to compare traditional recommendation approaches and understand how different recommendation strategies behave in real-world recommendation problems.

---

# Project Objectives

* Build recommendation systems using classical ML techniques
* Compare content-based vs collaborative filtering approaches
* Handle noisy or misspelled user input using fuzzy matching
* Evaluate recommendation quality using RMSE and MAE
* Visualize similarity distributions between books
* Create reusable and modular recommendation pipelines

---

# Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* Surprise
* matplotlib
* YAML configuration
* thefuzz (fuzzy matching)

---

# Recommendation Approaches

## 1. Content-Based Recommendation

The content-based recommender suggests books similar to a selected book based on textual features.

### Features Used

* Book title
* Author
* Metadata/content information

### Methodology

* TF-IDF Vectorization
* Cosine Similarity
* Fuzzy title matching for user-friendly search

### Workflow

1. Load and preprocess book metadata
2. Convert textual content into TF-IDF vectors
3. Compute cosine similarity scores
4. Return top-N similar books

### Additional Features

* Fuzzy matching for misspelled titles
* Similarity score visualization
* CSV export for recommendations

### Example Output

```bash
Top 5 recommendations similar to 'Harry Potter':

Book Title                      Author
--------------------------------------------
The Hobbit                      J.R.R Tolkien
Percy Jackson                   Rick Riordan
The Golden Compass              Philip Pullman
...
```

---

## 2. Collaborative Filtering

The collaborative filtering recommender predicts user preferences based on historical user-item interactions.

### Algorithms Implemented

* SVD (Matrix Factorization)
* KNNBasic (Neighborhood-based CF)

### Evaluation Metrics

* RMSE
* MAE

### Workflow

1. Split ratings dataset into train/test
2. Train collaborative filtering model
3. Predict unseen ratings
4. Generate personalized recommendations

### Recommendation Logic

* Identify unrated books for each user
* Predict rating scores
* Return top-N highest predicted books


# Key Learnings

* Difference between content-based and collaborative recommendation systems
* Importance of feature engineering in recommendation systems
* Trade-offs between memory usage and similarity computation
* Evaluation of recommendation quality using ranking metrics
* Handling noisy user input in real-world systems

---

# Future Improvements

* Hybrid Recommendation System
* Implicit feedback handling
* Vector database integration
* LLM-powered semantic recommendations
* FastAPI deployment
* Recommendation API endpoints
* Real-time inference pipeline

---

# How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Content-Based Recommendation

```bash
python content_based_recommendation.py
```

## Run Collaborative Filtering

```bash
python collaborative_filtering.py
```

---

# Sample Use Cases

* E-commerce recommendation engines
* OTT/movie recommendations
* Product recommendation systems
* Personalized content discovery
* Retail customer personalization

---

# Author

Shruti Bhosale

Applied Machine Learning | AI Engineering | Recommendation Systems | LLM Observability
