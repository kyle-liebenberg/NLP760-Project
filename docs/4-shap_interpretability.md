## 4. Explainable AI & SHAP Interpretability Model (`src/interpret_model.py`)

### Overview

An explainable AI (XAI) post-hoc interpretability module built on cooperative game theory principles. It extracts, quantifies, and visually maps how much individual isiZulu subwords, prefixes, and suffixes influence the non-linear classification decisions of the trained functional CNN-LSTM network.

### Important Technical Details

* **Agnostic Permutation Framework:** The pipeline leverages a model-agnostic `shap.Explainer` wrapper rather than framework-dependent deep learning gradient rules. This completely circumvents architectural version conflicts common in modern eager-execution environments (TensorFlow 2.16+ / Keras 3).
* **High-Evaluation Confidence Bounds:** Configured with an explicit execution parameter of `max_evals=1000`. This scales to satisfy the exact permutation requirement for your calculated 95th percentile sequence length ($2 \times 404 \text{ tokens} + 1 = 809 \text{ required steps}$), ensuring mathematically accurate feature assignments.

### What it Does

It captures raw text chunks from unseen evaluation sets, passes them to a black-box functional graph wrapper, measures output probability spikes over a baseline training distribution, and generates publication-grade attribution bar plots (`outputs/figures/shap_attribution_sample_<idx>.png`).

### Why it Does it (Linguistic & Academic Motivation)

Deep learning models are frequently criticized for being untranslatable "black boxes." In a low-resource linguistic study, showing a high accuracy score is not enough. You must prove *why* the model made its decision.

1. **Validating Morphological Learning:** By mapping subword attribution force, this module checks if the CNN-LSTM is actually extracting features from stylometric morphology (e.g., specific combinations of subject concords like `aba-`, `izi-`, or verbal roots like `-hamba`) or merely guessing based on data shortcuts or punctuation noise.
2. **Context-Aware Style Tracking:** Because it monitors the full recurrent sequence, it can reveal why the exact same token receives entirely different attribution forces depending on where it sits in a sentence structure, demonstrating context-aware sequence dependencies.

### How it Does it (Layer Mechanics & Math)

* **Model Abstraction:** It wraps the compiled Keras model inside a standard input-output Python function, passing integers directly as tensor primitives (`tf.convert_to_tensor`).
* **Background Profiling:** It caches a steady background slice of 30 training samples to calculate base marginal expectations ($E[f(x)]$).
* **Permutation Perturbation:** It loops over a text instance, iteratively masking tokens, and observes the direct change in the output probability for the *true target author's class node*.
* **Attribution Extraction:** It isolates the Shapley values across the first 25 valid tokens (filtering out trailing `[PAD]` sequences) to create readable visualizations. Red bars represent positive attribution force (elements that drove the model to choose that author), while blue bars represent negative force (elements that suppressed that choice).

### Libraries Used

* `shap`: For computing game-theoretic Shapley value attributions.
* `matplotlib.pyplot` & `seaborn`: Integrated using a non-interactive backend (`matplotlib.use("Agg")`) to generate and save distribution charts directly to disk without UI dependencies.
