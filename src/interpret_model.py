"""
interpret_model.py
------------------
Uses the SHAP (SHapley Additive exPlanations) library to explain predictions 
made by the trained functional CNN-LSTM model on isiZulu text chunks.

Outputs:
    outputs/figures/shap_attribution_sample_<idx>.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tokenizers import Tokenizer

# paths
SPLITS_DIR    = os.path.join("data", "splits")
TOKENIZER_DIR = os.path.join("outputs", "bpe_tokenizer")
METRICS_DIR   = os.path.join("outputs", "metrics")
FIGURES_DIR   = os.path.join("outputs", "figures")
MODELS_DIR    = "models"

def main():
    import shap
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("\n[SHAP] Loading models, tokenizer, and dataset splits...")
    
    # load tokenizer and classes
    tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    with open(os.path.join(SPLITS_DIR, "label_classes.json"), "r") as f:
        label_classes = json.load(f)
        
    X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))
    
    df_test = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    
    # load trained model weights
    model_path = os.path.join(MODELS_DIR, "cnn_lstm_best.keras")
    print(f"[SHAP] Loading trained network from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # define a prediction wrapper function
    def model_predict_wrapper(x_numpy):
        x_tensor = tf.convert_to_tensor(x_numpy, dtype=tf.int32)
        preds = model.predict(x_tensor, verbose=0)
        return preds

    print("[SHAP] Initializing model-agnostic SHAP Explainer...")
    background = X_train[:30]
    explainer = shap.Explainer(model_predict_wrapper, background)
    
    sample_indices = [0, 10, 20] 
    print(f"[SHAP] Calculating attributions for test indices: {sample_indices}...")
    
    for idx in sample_indices:
        test_instance = X_test[idx:idx+1]
        true_label_idx = y_test[idx]
        true_author = label_classes[true_label_idx]
        
        raw_text = df_test["text"].iloc[idx]
        encoded = tokenizer.encode(raw_text)
        tokens = encoded.tokens[:test_instance.shape[1]]
        
        pad_id = tokenizer.token_to_id("[PAD]")
        valid_length = np.sum(test_instance[0] != pad_id)
        
        # only first 25 tokens otherwise graph way too long
        display_len = min(valid_length, 25)
        clean_tokens = tokens[:display_len]
        
        print(f"   [+] Computing Shapley values for index {idx}...")
        shap_results = explainer(X_test[idx:idx+1], max_evals=1000)
        

        author_shap = shap_results.values[0, :display_len, true_label_idx]
        
        print(f"   [+] Generating visual plot for Author: {true_author}")
        
        plt.figure(figsize=(12, 6))
        colors = ['#E8593C' if val >= 0 else '#4A90D9' for val in author_shap]
        
        sns.barplot(x=list(range(len(clean_tokens))), y=author_shap, palette=colors)
        plt.xticks(list(range(len(clean_tokens))), clean_tokens, rotation=45, ha="right", fontsize=11)
        
        plt.xlabel("isiZulu BPE Subword Tokens", fontsize=12)
        plt.ylabel("SHAP Attribution Force", fontsize=12)
        plt.title(f"SHAP Feature Importance Map: {true_author}\n(Red = Drives Prediction | Blue = Suppresses Prediction)", fontsize=13)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        plt.tight_layout()
        output_fig = os.path.join(FIGURES_DIR, f"shap_attribution_sample_{idx}.png")
        plt.savefig(output_fig, dpi=150)
        plt.close()
        
    print(f"\n[+] SHAP visualization files successfully saved inside: {FIGURES_DIR}/")

if __name__ == "__main__":
    main()