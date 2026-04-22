from datasets import load_dataset
import pandas as pd

def main():
    print("Downloading and loading the dataset...")
    # Load the Izindaba-Tindzaba dataset from Hugging Face
    # We will load the isiZulu split (often denoted as 'zu' or just default if it's mixed)
    try:
        dataset = load_dataset("dsfsi/za-isizulu-siswati-news", "isizulu")
    except Exception as e:
        print(f"Could not load specific isiZulu split, loading default: {e}")
        dataset = load_dataset("dsfsi/za-isizulu-siswati-news")

    print("\n--- Dataset Structure ---")
    print(dataset)

    # Let's convert the 'train' split to a Pandas DataFrame for easier viewing
    train_data = dataset['train']
    df = pd.DataFrame(train_data)

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- First 3 Rows ---")
    # Display the first 3 rows to manually inspect the text
    for index, row in df.head(3).iterrows():
        print(f"\nRow {index + 1}:")
        for col in df.columns:
            # Truncate long text for easier reading in the terminal
            val = str(row[col])
            if len(val) > 100:
                val = val[:100] + "..."
            print(f"  {col}: {val}")

if __name__ == "__main__":
    main()