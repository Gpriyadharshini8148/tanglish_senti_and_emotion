from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Concatenate, GlobalMaxPooling1D
from transformers import AutoTokenizer, TFXLMRobertaModel, TFAlbertModel, AutoConfig
import pickle
import numpy as np
import pandas as pd
import re
import os
import threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Constants
MAX_LEN = 128
XLM_R_MODEL = "xlm-roberta-base"
INDIC_BERT_MODEL = "ai4bharat/indic-bert"
MODELS_DIR = "../models"
DATASET_PATH = "../data/corrected_tanglish_dataset.csv"

# Globals for models and resources
hybrid_model = None
xlm_tokenizer = None
indic_tokenizer = None
le_sentiment = None
le_emotion = None
training_status = "Idle"

def load_resources():
    global hybrid_model, xlm_tokenizer, indic_tokenizer, le_sentiment, le_emotion
    try:
        # Load label encoders
        if os.path.exists(os.path.join(MODELS_DIR, "sentiment_label_encoder.pkl")):
            with open(os.path.join(MODELS_DIR, "sentiment_label_encoder.pkl"), "rb") as f:
                le_sentiment = pickle.load(f)
        if os.path.exists(os.path.join(MODELS_DIR, "emotion_label_encoder.pkl")):
            with open(os.path.join(MODELS_DIR, "emotion_label_encoder.pkl"), "rb") as f:
                le_emotion = pickle.load(f)
        
        # Load tokenizers
        xlm_tokenizer = AutoTokenizer.from_pretrained(XLM_R_MODEL)
        indic_tokenizer = AutoTokenizer.from_pretrained(INDIC_BERT_MODEL)

        # Load Hybrid Model (if exists)
        model_path = os.path.join(MODELS_DIR, "hybrid_sentiment_emotion_model")
        if os.path.exists(model_path):
             # For models with custom transformer layers, it's often better to rebuild and load weights
             # but we'll try loading directly first. Custom objects might be needed.
             try:
                hybrid_model = tf.keras.models.load_model(model_path, custom_objects={
                    'TFXLMRobertaModel': TFXLMRobertaModel,
                    'TFAlbertModel': TFAlbertModel
                })
                print("Hybrid model loaded from disk.")
             except Exception as e:
                print(f"Direct load failed, model might need to be rebuilt: {e}")
                # We will handle rebuilding in the training part
                
        if hybrid_model:
            print("All resources loaded successfully.")
        else:
            print("Warning: Hybrid model not found. Please run /train.")
            
    except Exception as e:
        print(f"Error loading resources: {e}")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_model(num_sentiment_classes, num_emotion_classes):
    # Inputs for XLM-R
    xlm_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_ids")
    xlm_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_mask")
    
    # Inputs for Indic BERT (ALBERT based)
    indic_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_ids")
    indic_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_mask")

    # XLM-R Branch
    xlm_transformer = TFXLMRobertaModel.from_pretrained(XLM_R_MODEL)
    xlm_out = xlm_transformer(xlm_ids, attention_mask=xlm_mask)[0] # (batch, seq, 768)

    # Indic BERT Branch
    indic_transformer = TFAlbertModel.from_pretrained(INDIC_BERT_MODEL)
    indic_out = indic_transformer(indic_ids, attention_mask=indic_mask)[0] # (batch, seq, 768)

    # Concatenate representations
    combined = Concatenate()([xlm_out, indic_out]) # (batch, seq, 1536)

    # CNN Layer
    cnn_out = Conv1D(128, 3, activation='relu', padding='same')(combined)
    cnn_out = Dropout(0.3)(cnn_out)

    # LSTM Layer
    lstm_out = LSTM(128, return_sequences=False)(cnn_out)
    lstm_out = Dropout(0.3)(lstm_out)

    # Hidden dense
    dense_out = Dense(64, activation='relu')(lstm_out)

    # Two output heads: Sentiment and Emotion
    sentiment_output = Dense(num_sentiment_classes, activation='softmax', name='sentiment')(dense_out)
    emotion_output = Dense(num_emotion_classes, activation='softmax', name='emotion')(dense_out)

    model = Model(inputs=[xlm_ids, xlm_mask, indic_ids, indic_mask], 
                  outputs=[sentiment_output, emotion_output])
    
    # Freeze transformer layers to speed up training if memory is low, 
    # but for "training with" we'll leave them trainable or fine-tune.
    # xlm_transformer.trainable = False
    # indic_transformer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_worker():
    global hybrid_model, training_status, le_sentiment, le_emotion
    try:
        training_status = "Training Started..."
        if not os.path.exists(DATASET_PATH):
            training_status = f"Error: Dataset not found at {DATASET_PATH}"
            return

        df = pd.read_csv(DATASET_PATH)
        df['clean_text'] = df['text'].apply(preprocess_text)

        # Encode Targets
        le_sentiment = LabelEncoder()
        y_sent = tf.keras.utils.to_categorical(le_sentiment.fit_transform(df['category']))
        
        le_emotion = LabelEncoder()
        y_emo = tf.keras.utils.to_categorical(le_emotion.fit_transform(df['emotion']))

        # Save Label Encoders
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, "sentiment_label_encoder.pkl"), "wb") as f:
            pickle.dump(le_sentiment, f)
        with open(os.path.join(MODELS_DIR, "emotion_label_encoder.pkl"), "wb") as f:
            pickle.dump(le_emotion, f)

        # Tokenize
        xlm_tok = AutoTokenizer.from_pretrained(XLM_R_MODEL)
        indic_tok = AutoTokenizer.from_pretrained(INDIC_BERT_MODEL)

        def tokenize(texts, tokenizer):
            return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="tf")

        xlm_enc = tokenize(df['clean_text'], xlm_tok)
        indic_enc = tokenize(df['clean_text'], indic_tok)

        # Split Data
        indices = np.arange(len(df))
        train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

        train_inputs = {
            'xlm_ids': xlm_enc['input_ids'].numpy()[train_idx],
            'xlm_mask': xlm_enc['attention_mask'].numpy()[train_idx],
            'indic_ids': indic_enc['input_ids'].numpy()[train_idx],
            'indic_mask': indic_enc['attention_mask'].numpy()[train_idx]
        }
        val_inputs = {
            'xlm_ids': xlm_enc['input_ids'].numpy()[val_idx],
            'xlm_mask': xlm_enc['attention_mask'].numpy()[val_idx],
            'indic_ids': indic_enc['input_ids'].numpy()[val_idx],
            'indic_mask': indic_enc['attention_mask'].numpy()[val_idx]
        }
        
        train_targets = {'sentiment': y_sent[train_idx], 'emotion': y_emo[train_idx]}
        val_targets = {'sentiment': y_sent[val_idx], 'emotion': y_emo[val_idx]}

        # Build and Train
        hybrid_model = build_model(len(le_sentiment.classes_), len(le_emotion.classes_))
        training_status = "Model built. Training epochs..."
        
        hybrid_model.fit(
            train_inputs, train_targets,
            validation_data=(val_inputs, val_targets),
            epochs=3, # Low epochs for demonstration, can be increased
            batch_size=8 # Small batch size to avoid OOM
        )

        # Save
        hybrid_model.save(os.path.join(MODELS_DIR, "hybrid_sentiment_emotion_model"))
        training_status = "Training Complete. Model Saved."
        load_resources() # Reload to update globals

    except Exception as e:
        training_status = f"Training Failed: {str(e)}"
        print(f"Error in training: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global training_status
    if training_status.startswith("Training") and "Complete" not in training_status and "Failed" not in training_status:
        return jsonify({'status': 'Training already in progress', 'details': training_status})
    
    thread = threading.Thread(target=train_worker)
    thread.start()
    return jsonify({'status': 'Training started in background'})

@app.route('/status')
def status():
    return jsonify({'status': training_status})

@app.route('/predict', methods=['POST'])
def predict():
    if not hybrid_model:
        return jsonify({'error': 'Hybrid model not loaded. Please train first.'}), 503

    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    processed_text = preprocess_text(text)
    
    # Pre-tokenize for both models
    xlm_enc = xlm_tokenizer(processed_text, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors="tf")
    indic_enc = indic_tokenizer(processed_text, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors="tf")
    
    # Predict
    preds = hybrid_model.predict({
        'xlm_ids': xlm_enc['input_ids'],
        'xlm_mask': xlm_enc['attention_mask'],
        'indic_ids': indic_enc['input_ids'],
        'indic_mask': indic_enc['attention_mask']
    })
    
    sent_probs = preds[0][0]
    emo_probs = preds[1][0]
    
    sent_idx = np.argmax(sent_probs)
    emo_idx = np.argmax(emo_probs)
    
    sent_label = le_sentiment.inverse_transform([sent_idx])[0]
    sent_conf = float(sent_probs[sent_idx])
    
    emo_label = le_emotion.inverse_transform([emo_idx])[0]
    emo_conf = float(emo_probs[emo_idx])
    
    return jsonify({
        'sentiment': sent_label.strip(),
        'sentiment_confidence': round(sent_conf * 100, 2),
        'emotion': emo_label.strip(),
        'emotion_confidence': round(emo_conf * 100, 2)
    })

if __name__ == '__main__':
    load_resources()
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)

