"""
train_cnn_lstm.py
-----------------
Loads the tokenized splits, dynamically fetches hyperparameters from metadata,
trains the CNN-LSTM model, and evaluates it against the test set.

Outputs:
    models/cnn_lstm_best.keras
    outputs/metrics/dl_report.json
"""

import os
import json
import numpy as np

import tensorflow as tf
import keras

keras.utils.set_random_seed(19)
tf.config.experimental.enable_op_determinism()

from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from cnn_lstm_model import build_cnn_lstm

SPLITS_DIR  = os.path.join("data", "splits")
METRICS_DIR = os.path.join("outputs", "metrics")
MODELS_DIR  = "models"

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("\nLoading data and metadata...")
    
    # load tokenizer metadata
    with open(os.path.join(METRICS_DIR, "tokenizer_metadata.json"), "r") as f:
        meta = json.load(f)
        
    vocab_size = meta["actual_vocab"]
    max_seq_len = meta["max_sequence_length"]
    num_classes = meta["num_classes"]
    label_classes = meta["classes"]
    
    # load numpy arrays
    X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(SPLITS_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(SPLITS_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(SPLITS_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

    print(f"Data loaded! Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    # build model
    print("\nBuilding CNN-LSTM Architecture...")
    model = build_cnn_lstm(vocab_size=vocab_size, max_seq_length=max_seq_len, num_classes=num_classes)
    model.summary()

    # define callbacks
    model_save_path = os.path.join(MODELS_DIR, "cnn_lstm_best.keras")
    
    # extended patience to 8 so it gives the network breathing room to optimize
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    
    # scales learning rate down if validation progress stalls for 3 epochs
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    # train model
    print("\nStarting Training Loop...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40, 
        batch_size=32,       
        callbacks=[early_stopping, checkpoint, lr_scheduler],
        verbose=1
    )

    # evaluate on test set
    print("\n" + "=" * 60)
    print("DEEP LEARNING MODEL EVALUATION (TEST SET)")
    print("=" * 60)
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    f1        = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    accuracy  = accuracy_score(y_test, y_pred)
    
    print(f"F1 Macro  : {f1:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=label_classes, zero_division=0))

    # save results
    dl_results = {
        "f1_macro": round(f1, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "accuracy": round(accuracy, 4)
    }
    
    with open(os.path.join(METRICS_DIR, "dl_report.json"), "w") as f:
        json.dump(dl_results, f, indent=2)
        
    print(f"[+] Model saved to {model_save_path}")
    print(f"[+] Evaluation metrics saved to {METRICS_DIR}/dl_report.json")

if __name__ == "__main__":
    main()