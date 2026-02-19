import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Constants
DATASET_PATH = "../data/corrected_tanglish_dataset.csv"
MODELS_DIR = "../models"
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train():
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(df)} rows")
    
    # 1. Cleaning
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # 2. Prepare Data
    X = df['clean_text']
    y_sentiment = df['category'] # e.g. Positive, Negative
    y_emotion = df['emotion']    # e.g. joy, sadness

    # Split
    X_train, X_test, y_sent_train, y_sent_test, y_emo_train, y_emo_test = train_test_split(
        X, y_sentiment, y_emotion, test_size=0.1, random_state=42
    )

    # 3. Train Sentiment Model
    print("Training Sentiment Model (TF-IDF + Logistic Regression)...")
    sentiment_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0))
    ])
    sentiment_pipeline.fit(X_train, y_sent_train)
    print("Sentiment Accuracy:", sentiment_pipeline.score(X_test, y_sent_test))

    # 4. Train Emotion Model
    print("Training Emotion Model (TF-IDF + Logistic Regression)...")
    emotion_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0))
    ])
    emotion_pipeline.fit(X_train, y_emo_train)
    print("Emotion Accuracy:", emotion_pipeline.score(X_test, y_emo_test))

    # 5. Save Models
    print("Saving models to ../models/...")
    with open(os.path.join(MODELS_DIR, "simple_sentiment_model.pkl"), "wb") as f:
        pickle.dump(sentiment_pipeline, f)
    
    with open(os.path.join(MODELS_DIR, "simple_emotion_model.pkl"), "wb") as f:
        pickle.dump(emotion_pipeline, f)

    print("\nDONE! Simple models trained and saved.")
    print("Run app.py now to use these models.")

if __name__ == "__main__":
    train()
