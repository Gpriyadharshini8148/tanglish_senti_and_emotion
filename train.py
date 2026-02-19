
import tensorflow as tf
print("Importing tensorflow.keras...")
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Enable Mixed Precision to save memory
try:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled.")
except Exception as e:
    print(f"Could not enable mixed precision: {e}")

try:
    import tf_keras as keras
    print("Using tf_keras")
except ImportError:
    import tensorflow.keras as keras
    print("Using tensorflow.keras")

from tf_keras.models import Model
from tf_keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Concatenate
print("Importing transformers...")
from transformers import AutoTokenizer, TFAutoModel
print("Importing other libs...")
import pickle
import numpy as np
import pandas as pd
import re
import threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc

# Constants
MAX_LEN = 64 
BATCH_SIZE = 4
# Switching to Memory-Efficient Models
# albert-base-v2: English/Roman (15MB embedding RAM)
# microsoft/Multilingual-MiniLM-L12-H384: Multilingual (100MB embedding RAM, 384 hid dim) - fits in RAM
XLM_R_MODEL = "albert-base-v2" 
INDIC_BERT_MODEL = "microsoft/Multilingual-MiniLM-L12-H384" 
MODELS_DIR = "models"
DATASET_PATH = "data/corrected_tanglish_dataset.csv"

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def configure_gpu_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Memory Growth Enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(e)

class TextDataGenerator(keras.utils.Sequence):
    def __init__(self, texts, sentiment_labels, emotion_labels, batch_size, xlm_tokenizer, indic_tokenizer, max_len):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.emotion_labels = emotion_labels
        self.batch_size = batch_size
        self.xlm_tokenizer = xlm_tokenizer
        self.indic_tokenizer = indic_tokenizer
        self.max_len = max_len
        self.indices = np.arange(len(self.texts))

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_texts = [self.texts[i] for i in batch_indices]
        
        # Tokenize on the fly
        xlm_enc = self.xlm_tokenizer(batch_texts, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="tf")
        indic_enc = self.indic_tokenizer(batch_texts, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="tf")
        
        inputs = {
            'xlm_ids': xlm_enc['input_ids'],
            'xlm_mask': xlm_enc['attention_mask'],
            'indic_ids': indic_enc['input_ids'],
            'indic_mask': indic_enc['attention_mask']
        }
        
        targets = {
            'sentiment': self.sentiment_labels[batch_indices],
            'emotion': self.emotion_labels[batch_indices]
        }
        
        return inputs, targets
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def build_model(num_sentiment_classes, num_emotion_classes):
    # Inputs for DistilBERT
    xlm_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_ids")
    xlm_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_mask")
    
    # Inputs for Indic BERT
    indic_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_ids")
    indic_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_mask")

    # DistilBERT Branch
    print(f"Loading {XLM_R_MODEL}...")
    try:
        xlm_transformer = TFAutoModel.from_pretrained(XLM_R_MODEL)
    except Exception as e:
        print(f"Error loading {XLM_R_MODEL}: {e}. Trying simple load.")
        xlm_transformer = TFAutoModel.from_pretrained(XLM_R_MODEL, from_pt=True)

    xlm_out = xlm_transformer(xlm_ids, attention_mask=xlm_mask)[0] 

    # Indic BERT Branch
    print(f"Loading {INDIC_BERT_MODEL}...")
    try:
        indic_transformer = TFAutoModel.from_pretrained(INDIC_BERT_MODEL)
    except Exception as e:
        print(f"Error loading {INDIC_BERT_MODEL}: {e}. Trying simple load.")
        indic_transformer = TFAutoModel.from_pretrained(INDIC_BERT_MODEL, from_pt=True)

    indic_out = indic_transformer(indic_ids, attention_mask=indic_mask)[0] 

    # Concatenate representations
    combined = Concatenate()([xlm_out, indic_out]) 

    # CNN Layer
    cnn_out = Conv1D(128, 3, activation='relu', padding='same')(combined)
    cnn_out = Dropout(0.3)(cnn_out)

    # LSTM Layer
    lstm_out = LSTM(128, return_sequences=False)(cnn_out)
    lstm_out = Dropout(0.3)(lstm_out)

    # Hidden dense
    dense_out = Dense(64, activation='relu')(lstm_out)

    # Output heads
    sentiment_output = Dense(num_sentiment_classes, activation='softmax', dtype='float32', name='sentiment')(dense_out)
    emotion_output = Dense(num_emotion_classes, activation='softmax', dtype='float32', name='emotion')(dense_out)

    model = Model(inputs=[xlm_ids, xlm_mask, indic_ids, indic_mask], 
                  outputs=[sentiment_output, emotion_output])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    configure_gpu_memory()
    print("Starting training process with Optimized Hybrid Model...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded. {len(df)} rows.")
    
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Encode Targets
    print("Encoding targets...")
    le_sentiment = LabelEncoder()
    y_sent = keras.utils.to_categorical(le_sentiment.fit_transform(df['category']))
    
    le_emotion = LabelEncoder()
    y_emo = keras.utils.to_categorical(le_emotion.fit_transform(df['emotion']))

    # Save Label Encoders
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "sentiment_label_encoder.pkl"), "wb") as f:
        pickle.dump(le_sentiment, f)
    with open(os.path.join(MODELS_DIR, "emotion_label_encoder.pkl"), "wb") as f:
        pickle.dump(le_emotion, f)
    
    # Split Data
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, stratify=df['category'])
    
    # Prepare data for generator (raw text)
    train_texts = df['clean_text'].iloc[train_idx].tolist()
    val_texts = df['clean_text'].iloc[val_idx].tolist()
    
    train_sent_labels = y_sent[train_idx]
    train_emo_labels = y_emo[train_idx]
    val_sent_labels = y_sent[val_idx]
    val_emo_labels = y_emo[val_idx]

    # Tokenizers
    print("Loading tokenizers...")
    xlm_tok = AutoTokenizer.from_pretrained(XLM_R_MODEL)
    indic_tok = AutoTokenizer.from_pretrained(INDIC_BERT_MODEL)

    # Generators
    train_gen = TextDataGenerator(train_texts, train_sent_labels, train_emo_labels, BATCH_SIZE, xlm_tok, indic_tok, MAX_LEN)
    val_gen = TextDataGenerator(val_texts, val_sent_labels, val_emo_labels, BATCH_SIZE, xlm_tok, indic_tok, MAX_LEN)

    # Build and Train
    print("Building model...")
    gc.collect()
    
    hybrid_model = build_model(len(le_sentiment.classes_), len(le_emotion.classes_))
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=2, 
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "best_model.keras"),
        monitor='val_sentiment_accuracy', 
        save_best_only=True,
        verbose=1
    )

    print("Starting training...")
    hybrid_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    # Save Final Model
    save_path = os.path.join(MODELS_DIR, "hybrid_sentiment_emotion_model.keras")
    print(f"Saving final model to {save_path}...")
    hybrid_model.save(save_path)
    print("Training Complete. Models Saved.")

if __name__ == "__main__":
    main()
