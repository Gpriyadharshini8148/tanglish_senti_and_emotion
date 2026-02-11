import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical

# Ensure models directory exists
if not os.path.exists("../models"):
    os.makedirs("../models")

# Load dataset
data_path = "../tanglish_sentiment_emotion_dataset.tsv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit()

print("Loading dataset...")
df = pd.read_csv(data_path, sep='\t')
print("Dataset Shape:", df.shape)

# Preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Tokenization
MAX_WORDS = 20000
MAX_LEN = 100

print("Tokenizing text...")
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df['clean_text'])

X = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X, maxlen=MAX_LEN)

# Save tokenizer
with open("../models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved.")

def create_model(output_units, input_dim=MAX_WORDS, input_len=MAX_LEN):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=input_len))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_units, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- Train Sentiment Model ---
print("\nTraining Sentiment Model...")
y_sentiment = df['category'].values
le_sentiment = LabelEncoder()
y_sent_encoded = le_sentiment.fit_transform(y_sentiment)
y_sent_onehot = to_categorical(y_sent_encoded)

# Save Mapping
sentiment_map = dict(zip(le_sentiment.classes_, le_sentiment.transform(le_sentiment.classes_)))
print("Sentiment Map:", sentiment_map)
with open("../models/sentiment_label_encoder.pkl", "wb") as f:
    pickle.dump(le_sentiment, f)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y_sent_onehot, test_size=0.2, random_state=42)

# Train
sentiment_model = create_model(output_units=y_sent_onehot.shape[1])
sentiment_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)

# Save Model
sentiment_model.save("../models/sentiment_model.h5")
print("Sentiment Model Saved.")

# --- Train Emotion Model ---
print("\nTraining Emotion Model...")
y_emotion = df['emotion'].values
le_emotion = LabelEncoder()
y_emo_encoded = le_emotion.fit_transform(y_emotion)
y_emo_onehot = to_categorical(y_emo_encoded)

# Save Mapping
emotion_map = dict(zip(le_emotion.classes_, le_emotion.transform(le_emotion.classes_)))
print("Emotion Map:", emotion_map)
with open("../models/emotion_label_encoder.pkl", "wb") as f:
    pickle.dump(le_emotion, f)

# Split
X_train_emo, X_val_emo, y_train_emo, y_val_emo = train_test_split(X, y_emo_onehot, test_size=0.2, random_state=42)

# Train
emotion_model = create_model(output_units=y_emo_onehot.shape[1])
emotion_model.fit(X_train_emo, y_train_emo, validation_data=(X_val_emo, y_val_emo), epochs=5, batch_size=64)

# Save Model
emotion_model.save("../models/emotion_model.h5")
print("Emotion Model Saved.")
