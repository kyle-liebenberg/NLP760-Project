import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    # 1. Define Paths
    train_path = 'data/processed/train_data.csv'
    test_path = 'data/processed/test_data.csv'

    print("Loading train and test datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Separate features and labels
    X_train_raw = df_train['text'].fillna('')
    y_train = df_train['label']
    X_test_raw = df_test['text'].fillna('')
    y_test = df_test['label']

    # 2. TF-IDF Vectorization
    print("Vectorizing text using TF-IDF (Character n-grams 2-5)...")
    # Using char n-grams is highly effective for agglutinative languages like isiZulu
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=10000)
    
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    # 3. Model 1: Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_preds = lr_model.predict(X_test_tfidf)
    
    evaluate_model("Logistic Regression", y_test, lr_preds)

    # 4. Model 2: Support Vector Machine (LinearSVC)
    print("\n--- Training Support Vector Machine (LinearSVC) ---")
    svm_model = LinearSVC(random_state=42, max_iter=2000)
    svm_model.fit(X_train_tfidf, y_train)
    svm_preds = svm_model.predict(X_test_tfidf)
    
    evaluate_model("Support Vector Machine", y_test, svm_preds)

def evaluate_model(model_name, y_true, y_pred):
    """Calculates and prints the evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    # Using 'weighted' average to account for any slight class imbalances among the 11 authors
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Results for {model_name}:")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")

if __name__ == "__main__":
    main()