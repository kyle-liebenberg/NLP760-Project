import pandas as pd
import numpy as np
import os
import pickle
from tokenizers import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    # 1. Define Paths
    csv_path = 'data/raw/isizulu_authors_dataset.csv'
    tokenizer_path = 'models/isizulu_bpe_tokenizer.json'
    processed_dir = 'data/processed'
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Loading tokenizer and dataset...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing text or authors just in case
    df = df.dropna(subset=['text', 'author'])
    
    # 2. Encode Labels
    print("Encoding author labels...")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['author'])
    
    # Save the label encoder mapping for evaluation and SHAP interpretation later
    with open(os.path.join(processed_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
        
    # 3. Train/Test Split
    # We stratify by 'label' to ensure each author is equally represented in both sets
    print("Splitting dataset into train and test sets...")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Save raw splits for TF-IDF Baselines
    df_train.to_csv(os.path.join(processed_dir, 'train_data.csv'), index=False)
    df_test.to_csv(os.path.join(processed_dir, 'test_data.csv'), index=False)
    
    # 4. Tokenization and Padding (for CNN-LSTM)
    MAX_LEN = 500
    pad_id = tokenizer.token_to_id("[PAD]")
    
    print("Tokenizing and padding sequences for deep learning...")
    
    # Process Training Set
    train_sequences = [tokenizer.encode(text).ids for text in df_train['text'].tolist()]
    X_train = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post', value=pad_id)
    y_train = df_train['label'].values
    
    # Process Testing Set
    test_sequences = [tokenizer.encode(text).ids for text in df_test['text'].tolist()]
    X_test = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post', value=pad_id)
    y_test = df_test['label'].values
    
    # 5. Save Processed Artifacts
    print("Saving NumPy arrays...")
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    
    print("Preprocessing complete! All artifacts saved in data/processed/")

if __name__ == "__main__":
    main()