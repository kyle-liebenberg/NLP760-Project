"""
baselines.py
------------
Trains and evaluates traditional NLP baselines (BoW + TF-IDF with SVM
and Logistic Regression) on the isiZulu authorship attribution dataset.

This establishes the score that Diya's CNN-LSTM must beat in Sprint 2.

Usage (from project root):
    python src/baselines.py

Outputs:
    outputs/metrics/baseline_report.json
    outputs/figures/baseline_comparison.png
    outputs/figures/confusion_matrix_<model>.png  (for best model)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)

# ── Paths ────────────────────────────────────────────────────────────────────
SPLITS_DIR   = os.path.join("data", "splits")
FIGURES_DIR  = os.path.join("outputs", "figures")
METRICS_DIR  = os.path.join("outputs", "metrics")
RANDOM_STATE = 42
MAX_FEATURES = 10000

# Load the exact train/test partitions used throughout the project
def load_splits():
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv")) 
    test_df  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    print(f"[load]  train={len(train_df)}, test={len(test_df)}, val={len(val_df)}")
    return train_df, test_df 

# Construct multiple traditional NLP baselines so the CNN-LSTM
# can later be compared against both lexical and morphological models.
def build_pipelines() -> dict:
    """
    Five baseline pipelines covering all combinations from the proposal:
      - BoW  (word n-grams) + SVM
      - BoW  (char n-grams) + SVM
      - TF-IDF (word n-grams) + SVM       ← typically the strongest
      - TF-IDF (char n-grams) + SVM
            Captures subword morphology and stylistic spelling patterns,
            which are especially important for agglutinative isiZulu text.
      - TF-IDF (word n-grams) + LogReg    ← for comparison
    """
    return {
        "BoW_word_SVM": Pipeline([
            ("vec", CountVectorizer(analyzer="word", ngram_range=(1, 2),
                                    max_features=MAX_FEATURES)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "BoW_char_SVM": Pipeline([
            ("vec", CountVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                    max_features=MAX_FEATURES)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "TFIDF_word_SVM": Pipeline([
            ("vec", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     sublinear_tf=True, max_features=MAX_FEATURES)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "TFIDF_char_SVM": Pipeline([
            ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                     sublinear_tf=True, max_features=MAX_FEATURES)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "TFIDF_word_LogReg": Pipeline([
            ("vec", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     sublinear_tf=True, max_features=MAX_FEATURES)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE,
                                        solver="lbfgs")),
        ]),
    }
# Train and evaluate every baseline model under the same
# conditions to produce fair benchmark comparisons.

def evaluate_all(pipelines: dict, X_train, y_train, X_test, y_test,
                 label_names: list) -> dict:
    results = {}
    print("\n" + "=" * 65)
    print("BASELINE RESULTS")
    print("=" * 65)

    assert len(X_train) == len(y_train), \
    "Training texts and labels must have matching lengths"

    assert len(X_test) == len(y_test), \
        "Test texts and labels must have matching lengths"

    assert len(label_names) > 1, \
        "At least two author classes are required"
    
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        assert len(preds) == len(y_test), \
        f"{name} produced incorrect prediction count"

        f1        = f1_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro", zero_division=0)
        recall    = recall_score(y_test, preds, average="macro", zero_division=0)
        accuracy  = accuracy_score(y_test, preds)
        report    = classification_report(y_test, preds,
                                          target_names=label_names,
                                          output_dict=True)

        results[name] = {
            "f1_macro":       round(f1, 4),
            "precision_macro": round(precision, 4),
            "recall_macro":   round(recall, 4),
            "accuracy":       round(accuracy, 4),
            "per_class":      report,
        }

        # Macro F1 is the primary metric because the dataset is mildly
        # imbalanced and every author should contribute equally to evaluation.
        print(f"\n── {name} ──")
        print(f"   F1 macro : {f1:.4f}")
        print(f"   Accuracy : {accuracy:.4f}")
        print(classification_report(y_test, preds, target_names=label_names,
                                    zero_division=0))

    return results


def plot_comparison(results: dict, figures_dir: str):
    """Bar chart comparing all baselines on F1 macro."""
    os.makedirs(figures_dir, exist_ok=True)
    names  = list(results.keys())
    f1s    = [results[n]["f1_macro"] for n in names]
    accs   = [results[n]["accuracy"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, f1s, width, label="F1 macro", color="#4A90D9",
                    alpha=0.85)
    bars2 = ax.bar(x + width/2, accs, width, label="Accuracy", color="#E8593C",
                    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Baseline model comparison — isiZulu authorship attribution",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(figures_dir, "baseline_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n✓  Comparison chart saved → {out}")


def plot_confusion_matrix(pipeline, X_test, y_test, label_names: list,
                          model_name: str, figures_dir: str):
    """Confusion matrix heatmap for the best-performing model."""
    preds = pipeline.predict(X_test)
    cm    = confusion_matrix(y_test, preds , normalize="true")
    short = [a.split()[-1] for a in label_names]   # last name only for brevity

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=short, yticklabels=short,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted author", fontsize=12)
    ax.set_ylabel("True author", fontsize=12)
    ax.set_title(f"Confusion matrix — {model_name}", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    out   = os.path.join(figures_dir, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✓  Confusion matrix saved → {out}")


def save_results(results: dict, metrics_dir: str, X_train, X_test):
    os.makedirs(metrics_dir, exist_ok=True)

    metadata = {
        "random_state": RANDOM_STATE,
        "max_features": MAX_FEATURES,
        "dataset_sizes": {
            "train": len(X_train),
            "test": len(X_test),
        },
        "models": list(results.keys()),
    }

    report_path = os.path.join(metrics_dir, "baseline_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓  Baseline report saved → {report_path}")

    metadata_path = os.path.join(metrics_dir, "baseline_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓  Baseline metadata saved → {metadata_path}")


def print_summary_table(results: dict):
    print("\n" + "=" * 65)
    print("SUMMARY TABLE  (paste this into your README)")
    print("=" * 65)
    print(f"{'Model':<22} {'F1 Macro':>9} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}")
    print("-" * 65)
    best_f1 = 0
    best_name = ""
    for name, r in results.items():
        marker = " ← BEST" if r["f1_macro"] == max(
            v["f1_macro"] for v in results.values()) else ""
        print(f"{name:<22} {r['f1_macro']:>9.4f} {r['precision_macro']:>10.4f} "
              f"{r['recall_macro']:>8.4f} {r['accuracy']:>10.4f}{marker}")
        if r["f1_macro"] > best_f1:
            best_f1  = r["f1_macro"]
            best_name = name
    print("=" * 65)
    print(f"\n🏆  Best baseline: {best_name}  (F1 macro = {best_f1:.4f})")
    print(f"    CNN-LSTM (Sprint 2) must exceed F1 = {best_f1:.4f} to justify the approach.")
    return best_name


def run():
    train_df, test_df = load_splits()
    X_train, y_train = train_df["text"], train_df["author"]
    X_test,  y_test  = test_df["text"],  test_df["author"]
    label_names = sorted(train_df["author"].unique().tolist())

    pipelines = build_pipelines()
    results   = evaluate_all(pipelines, X_train, y_train, X_test, y_test,
                              label_names)

    best_name = print_summary_table(results)
    plot_comparison(results, FIGURES_DIR)
    plot_confusion_matrix(pipelines[best_name], X_test, y_test, label_names,
                          best_name, FIGURES_DIR)
    save_results(results, METRICS_DIR, X_train, X_test)
    return results


if __name__ == "__main__":
    run()