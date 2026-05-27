"""
interpret_model.py
------------------
Uses the SHAP (SHapley Additive exPlanations) library to explain predictions 
made by the trained functional CNN-LSTM model on isiZulu text chunks.

Bypasses TensorFlow 1.x session dependencies by utilizing a model-agnostic 
SHAP Explainer wrapper.

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

# ── Paths ──────────────────────────────────────────────────
SPLITS_DIR    = os.path.join("data", "splits")
TOKENIZER_DIR = os.path.join("outputs", "bpe_tokenizer")
METRICS_DIR   = os.path.join("outputs", "metrics")
FIGURES_DIR   = os.path.join("outputs", "figures")
MODELS_DIR    = "models"

def main():
    import shap
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("\n[SHAP] Loading models, tokenizer, and dataset splits...")
    
    # 1. Load Tokenizer & Classes
    tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    with open(os.path.join(SPLITS_DIR, "label_classes.json"), "r") as f:
        label_classes = json.load(f)
        
    # 2. Load Processed Arrays
    X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))
    
    # Load original test dataframe to see the raw text mappings
    df_test = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    
    # 3. Load Trained Model Weights
    model_path = os.path.join(MODELS_DIR, "cnn_lstm_best.keras")
    print(f"[SHAP] Loading trained network from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Define a Prediction Wrapper Function
    # This prevents SHAP from seeking legacy backend sessions by abstracting
    # the Keras model into a standard input-output numpy function.
    def model_predict_wrapper(x_numpy):
        # Cast input back to integer tensors since BPE token IDs are integers
        x_tensor = tf.convert_to_tensor(x_numpy, dtype=tf.int32)
        preds = model.predict(x_tensor, verbose=0)
        return preds

    # 5. Initialize the Model-Agnostic Explainer
    # We pass a background set of 30 samples to serve as a baseline reference distribution.
    print("[SHAP] Initializing model-agnostic SHAP Explainer...")
    background = X_train[:30]
    explainer = shap.Explainer(model_predict_wrapper, background)
    
    # 6. Select Sample Articles for Visual Analysis
    sample_indices = [0, 10, 20] # Distinct test indices to inspect
    print(f"[SHAP] Calculating attributions for test indices: {sample_indices}...")
    
    for idx in sample_indices:
        test_instance = X_test[idx:idx+1]
        true_label_idx = y_test[idx]
        true_author = label_classes[true_label_idx]
        
        # Extract and reconstruct human-readable tokens
        raw_text = df_test["text"].iloc[idx]
        encoded = tokenizer.encode(raw_text)
        tokens = encoded.tokens[:test_instance.shape[1]]
        
        # Reconstruct token-to-length boundaries (clip excessive padding noise)
        pad_id = tokenizer.token_to_id("[PAD]")
        valid_length = np.sum(test_instance[0] != pad_id)
        
        # Limit to the first 25 valid tokens to create highly readable bar plots
        display_len = min(valid_length, 25)
        clean_tokens = tokens[:display_len]
        
        # Compute Shapley values
        print(f"   [+] Computing Shapley values for index {idx}...")
        shap_results = explainer(X_test[idx:idx+1], max_evals=1000)
        
        # Extract attributions corresponding specifically to the True Author's class node
        # shape profile is: [num_instances, max_seq_length, num_classes]
        author_shap = shap_results.values[0, :display_len, true_label_idx]
        
        # 7. Generate Attribution Bar Chart
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
        
    print(f"\n✓ Phase 5 Complete! SHAP visualization files successfully saved inside: {FIGURES_DIR}/")

if __name__ == "__main__":
    main()