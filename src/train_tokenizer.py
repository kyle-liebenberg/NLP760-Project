import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

def main():
    # 1. Define paths relative to the root directory
    csv_path = 'data/raw/isizulu_authors_dataset.csv'
    model_dir = 'models'
    
    # Ensure the models directory exists
    os.makedirs(model_dir, exist_ok=True)

    # 2. Extract text for training
    print("Loading isiZulu dataset...")
    df = pd.read_csv(csv_path)
    
    # Drop any empty rows and convert the text column to a list
    texts = df['text'].dropna().tolist()

    # Write to a temporary text file (HuggingFace tokenizers train directly from text files)
    temp_file = 'temp_training_text.txt'
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    # 3. Initialize and train the BPE Tokenizer
    print("Training BPE Tokenizer on isiZulu morphology...")
    
    # Initialize tokenizer with an unknown token for words it hasn't seen
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Pre-tokenize by splitting on whitespace
    tokenizer.pre_tokenizer = Whitespace()

    # Configure the trainer with special tokens our neural network will need later
    trainer = BpeTrainer(
        vocab_size=5000, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    # Train the tokenizer
    tokenizer.train(files=[temp_file], trainer=trainer)

    # 4. Save the tokenizer for the preprocessing phase
    save_path = os.path.join(model_dir, 'isizulu_bpe_tokenizer.json')
    tokenizer.save(save_path)
    
    print(f"Success! Tokenizer trained and saved to: {save_path}")

    # Clean up the temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    main()