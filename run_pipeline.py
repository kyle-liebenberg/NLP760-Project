"""
run_pipeline.py
---------------
Master orchestration script for the isiZulu Authorship Attribution Project.
Sequentially runs preprocessing, tokenization, baselines, deep learning, 
and interpretability modules while outputting summary report cards.

Usage:
    python run_pipeline.py
"""

import os
import sys
import time
import subprocess

def print_header(title):
    print("\n" + "=" * 70)
    print(f" >>> PHASE: {title.upper()} <<<")
    print("=" * 70)

def run_script(script_path):
    """Executes a Python script as a subprocess and streams its output."""
    start_time = time.time()
    
    # Run using the current virtual environment environment interpreter
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output to terminal in real-time
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    elapsed = time.time() - start_time
    
    if process.returncode != 0:
        print(f"\n[!] CRITICAL ERROR: {script_path} failed with exit code {process.returncode}")
        sys.exit(1)
        
    return elapsed

def display_summary(times):
    print("\n" + "=" * 70)
    print("   MASTER EXECUTION COMPLETE — SPRINT SYSTEM SUCCESS")
    print("=" * 70)
    print(f"{'Pipeline Component':<35} | {'Execution Time':<15}")
    print("-" * 70)
    for component, duration in times.items():
        print(f"{component:<35} | {duration:.2f} seconds")
    print("-" * 70)
    print(f"{'TOTAL PIPELINE WALL TIME':<35} | {sum(times.values()):.2f} seconds")
    print("=" * 70)
    print("✓ All metrics saved to outputs/metrics/")
    print("✓ All visual plots saved to outputs/figures/\n")

def main():
    # Verify we are running from the root directory
    if not os.path.exists(os.path.join("src", "preprocessing.py")):
        print("[!] Execution Error: Please run run_pipeline.py from the project root directory.")
        sys.exit(1)
        
    execution_times = {}

    # Phase 1: Preprocessing & Data Augmentation Chunking
    print_header("1. Preprocessing & Data Augmentation")
    t_prep = run_script(os.path.join("src", "preprocessing.py"))
    execution_times["Data Augmentation & Splits"] = t_prep

    # Phase 2: Subword Tokenizer Training & Array Sequence Generation
    print_header("2. BPE Subword Tokenizer Training")
    t_tok = run_script(os.path.join("src", "tokenizer.py"))
    execution_times["Tokenizer Training & Encoding"] = t_tok

    # Phase 3: Traditional NLP Lexical & Morphological Baselines
    print_header("3. Traditional NLP Baselines (TF-IDF vs BoW)")
    t_base = run_script(os.path.join("src", "baselines.py"))
    execution_times["Lexical Baseline Training"] = t_base

    # Phase 4: Hybrid CNN-LSTM Neural Network Optimization
    print_header("4. Hybrid CNN-LSTM Optimization Network")
    t_dl = run_script(os.path.join("src", "train_cnn_lstm.py"))
    execution_times["Deep Learning Training Loop"] = t_dl

    # Phase 5: Post-Hoc Model Interpretability & XAI SHAP Mapping
    print_header("5. Explainable AI SHAP Attribution Analysis")
    t_shap = run_script(os.path.join("src", "interpret_model.py"))
    execution_times["SHAP Feature Interpretability"] = t_shap

    # Display Master Reporting Block
    display_summary(execution_times)

if __name__ == "__main__":
    main()