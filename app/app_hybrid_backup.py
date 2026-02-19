import os
# Force legacy keras for compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

try:
    import tf_keras as keras
except ImportError:
    import tensorflow.keras as keras

from tf_keras.models import Model, load_model
from tf_keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Concatenate, GlobalMaxPooling1D
from transformers import AutoTokenizer, TFAutoModel, AutoConfig
import pickle
import numpy as np
import pandas as pd
import re
import os
import threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys

app = Flask(__name__, 
            static_folder='../frontend/dist/assets',
            template_folder='../frontend/dist',
            static_url_path='/assets')
CORS(app) # Enable CORS for all routes (still useful for dev)

# Constants
MAX_LEN = 64
XLM_R_MODEL = "xlm-roberta-base"
INDIC_BERT_MODEL = "microsoft/Multilingual-MiniLM-L12-H384"
MODELS_DIR = "../model"
DATASET_PATH = "../data/corrected_tanglish_dataset.csv"

# Globals for models and resources
hybrid_model = None
xlm_tokenizer = None
indic_tokenizer = None
le_sentiment = None
le_emotion = None
training_status = "Idle"

def build_model(num_sentiment_classes, num_emotion_classes):
    # Inputs for XLM-R
    xlm_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_ids")
    xlm_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="xlm_mask")
    
    # Inputs for Indic BERT (ALBERT based)
    indic_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_ids")
    indic_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="indic_mask")

    # XLM-R Branch
    from transformers import TFAutoModel
    try:
        xlm_transformer = TFAutoModel.from_pretrained(XLM_R_MODEL)
    except:
        xlm_transformer = TFAutoModel.from_pretrained(XLM_R_MODEL, from_pt=True)

    # Indic BERT Branch
    try:
        indic_transformer = TFAutoModel.from_pretrained(INDIC_BERT_MODEL)
    except:
        indic_transformer = TFAutoModel.from_pretrained(INDIC_BERT_MODEL, from_pt=True)

    xlm_out = xlm_transformer(xlm_ids, attention_mask=xlm_mask)[0] 
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

    # Two output heads: Sentiment and Emotion
    sentiment_output = Dense(num_sentiment_classes, activation='softmax', name='sentiment')(dense_out)
    emotion_output = Dense(num_emotion_classes, activation='softmax', name='emotion')(dense_out)

    model = Model(inputs=[xlm_ids, xlm_mask, indic_ids, indic_mask], 
                  outputs=[sentiment_output, emotion_output])
    return model

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
        print("Loading tokenizers...")
        xlm_tokenizer = AutoTokenizer.from_pretrained(XLM_R_MODEL)
        indic_tokenizer = AutoTokenizer.from_pretrained(INDIC_BERT_MODEL)

        # Load Hybrid Model (if exists)
        model_path = os.path.join(MODELS_DIR, "hybrid_sentiment_emotion_model.keras")
        if os.path.exists(model_path):
             try:
                print(f"Building model architecture...")
                # Rebuild architecture first
                hybrid_model = build_model(len(le_sentiment.classes_), len(le_emotion.classes_))
                
                print(f"Loading weights from {model_path}...")
                hybrid_model.load_weights(model_path, skip_mismatch=True)
                print("Hybrid model loaded from disk.")
             except Exception as e:
                print(f"Weight load failed: {e}")
                import traceback
                traceback.print_exc()
                
        if hybrid_model:
            print("All resources loaded successfully.")
        else:
            print("Warning: Hybrid model not found. Please verify models folder.")
            
    except Exception as e:
        print(f"Error loading resources: {e}")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
        
        hybrid_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

        hybrid_model.fit(
            train_inputs, train_targets,
            validation_data=(val_inputs, val_targets),
            epochs=3, # Low epochs for demonstration, can be increased
            batch_size=8 # Small batch size to avoid OOM
        )

        # Save
        hybrid_model.save(os.path.join(MODELS_DIR, "hybrid_sentiment_emotion_model.keras"))
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
    global hybrid_model
    if not hybrid_model:
        print("Model not found in global scope, attempting to reload...")
        load_resources()
        
    if not hybrid_model:
        return jsonify({'error': 'Hybrid model not loaded. Please train first.'}), 503

    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    processed_text = preprocess_text(text)
    print(f"DEBUG: Processing: '{text}' -> '{processed_text}'")
    
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
    
    # Debug Probabilities - DETAILED OUTPUT
    if le_sentiment:
        sent_dict = dict(zip(le_sentiment.classes_, sent_probs))
        # Sort by probability
        sorted_sent = dict(sorted(sent_dict.items(), key=lambda item: item[1], reverse=True))
        print("\n--- SENTIMENT PROBABILITIES ---")
        for k, v in sorted_sent.items():
            print(f"  {k}: {v:.4f}")
            
    if le_emotion:
        emo_dict = dict(zip(le_emotion.classes_, emo_probs))
        sorted_emo = dict(sorted(emo_dict.items(), key=lambda item: item[1], reverse=True))
        print("\n--- EMOTION PROBABILITIES ---")
        for k, v in sorted_emo.items():
            print(f"  {k}: {v:.4f}")
    print("---------------------------------\n", flush=True)
    
    # --- Smart Post-Processing ---
    # Get indices sorted by probability (descending)
    sent_indices_sorted = np.argsort(sent_probs)[::-1]
    emo_indices_sorted = np.argsort(emo_probs)[::-1]
    
    top_sent_idx = sent_indices_sorted[0]
    top_sent_label = le_sentiment.inverse_transform([top_sent_idx])[0].strip()
    top_sent_conf = float(sent_probs[top_sent_idx])
    
    # If top prediction is 'not-Tamil' or 'unknown_state' but confidence is low (< 0.6), look for next best
    final_sent_label = top_sent_label
    final_sent_conf = top_sent_conf
    


    # Emotion logic (similar)
    top_emo_idx = emo_indices_sorted[0]
    final_emo_label = le_emotion.inverse_transform([top_emo_idx])[0].strip()
    final_emo_conf = float(emo_probs[top_emo_idx])
    
    # Generate Insight / Decision
    insight = generate_insight(final_sent_label, final_emo_label, final_sent_conf)

    return jsonify({
        'sentiment': final_sent_label,
        'sentiment_confidence': round(final_sent_conf * 100, 2),
        'emotion': final_emo_label,
        'emotion_confidence': round(final_emo_conf * 100, 2),
        'decision': insight
    })

def generate_insight(sentiment, emotion, confidence):
    s = sentiment.lower()
    e = emotion.lower()
    
    if "positive" in s:
        if "love" in e or "joy" in e:
            return "✅ User is highly satisfied! This is excellent feedback. Recommend highlighting this testimonial or thanking the user for their positive support."
        elif "trust" in e:
             return "✅ User expresses trust in the service. Maintain this relationship by delivering consistent quality."
        else:
             return "✅ Positive feedback detected. The user had a good experience."
             
    elif "negative" in s:
        if "anger" in e:
            return "⚠️ CRITICAL: User is angry. Immediate escalation recommended. Apologize and offer a resolution to prevent churn."
        elif "disgust" in e:
            return "⚠️ Negative feedback on quality/standards. Investigate the product/service aspects mentioned immediately."
        elif "fear" in e:
            return "⚠️ User expresses concern or fear. Reassure them about safety/reliability policies."
        elif "sadness" in e:
            return "⚠️ User had a disappointing experience. Reach out to understand what went wrong and offer compensation."
            
    elif "mixed" in s:
        return "Thinking... User has mixed feelings. They see both pros and cons. Recommend analyzing specific keywords to improve the weak areas."
        
    elif "not-tamil" in s or "unknown" in s:
         return "❓ Dataset/Model Uncertainty: The text might be out of domain or insufficient context. Verify if the input is valid Tanglish."
         
    return "Analyzing sentiment patterns..."

# Load resources immediately on module import
try:
    load_resources()
except Exception as e:
    print(f"Initial resource load failed: {e}")

if __name__ == '__main__':
    if le_sentiment:
        print(f"DEBUG: Sentiment Classes: {len(le_sentiment.classes_)} ({le_sentiment.classes_})")
    if le_emotion:
        print(f"DEBUG: Emotion Classes: {len(le_emotion.classes_)} ({le_emotion.classes_})")
    print(f"DEBUG: Global hybrid_model is: {hybrid_model}")
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)
