# NLP760-Project
This project applies subword embeddings to authorship attribution, aiming to identify an author’s distinctive writing style by analysing subword patterns rather than whole words.

---
# The Process:

## 1. Data Scraping 
isiZulu articles were scraped from Isolezwe (`https://isolezwe.co.za`) for the following 11 authors
- Mhlengi Shangase: `https://isolezwe.co.za/authors/journalist/`
- Zimbili Vilakazi: `https://isolezwe.co.za/authors/zee/`
- Sabelo Nsele: `https://isolezwe.co.za/authors/mangethe/`
- Simangaliso Ntshangase: `https://isolezwe.co.za/authors/health-and-lifestyle-journalist/`
- Fanelesibonge Bengu: `https://isolezwe.co.za/authors/fanele-bengu/`
- Nonkululeko Nhlapo: `https://isolezwe.co.za/authors/online/`
- Nokubongwa Phenyane: `https://isolezwe.co.za/authors/maphenyane/`
- Zakhele Xaba: `https://isolezwe.co.za/authors/zakhele-xaba/`
- Sibusiso Mdlalose: `https://isolezwe.co.za/authors/mdlalose-wezemidlalo/`
- Mthokozisi Mncuseni: `https://isolezwe.co.za/authors/your-football-guy/`
- Charles Khuzwayo: `https://isolezwe.co.za/authors/entertainment`

## To run the python script:
```bash
python3 -m venv venv
source venv/bin/activate
pip install datasets pandas
pip install requests beautifulsoup4
python3 src/build_author_dataset.py
```

We decided to use data from Isolezwe because existing datasets out there did not contain enough data or split the data between authors.
This results in a total of 261 samples (+-70 000 words) of isiZulu text across 11 authors. 
Although not a huge dataset, the purpose of this project is to work with low-resource african languages. 

## 2. Preprocessing and Baselines
### 2.1. Tokeniser
 It reads the isiZulu articles, extracts all the text into a temporary file, and feeds it to the Byte-Pair Encoding (BPE) algorithm. The algorithm will scan the text, merge the most frequent character pairs, and learn the common isiZulu prefixes and suffixes (like uku-, ama-, nga-). Finally, it sets a maximum vocabulary size of 5000 (a sweet spot for small datasets to prevent data sparsity) and saves the trained model.

To run the script:
```bash
pip install pandas tokenizers
python3 src/train_tokenizer.py
```

### 2.2. Preprocessing 
- Load the trained tokenizer and your CSV.
- Encode the 11 author names into numerical labels (0-10) using LabelEncoder.
- Split the raw data into Training (80%) and Testing (20%) sets and save them as CSVs (for the TF-IDF baselines).
- Tokenize the texts into integer sequences.
- Pad those sequences to a maximum length of 500 tokens (for the Keras CNN-LSTM) and save them as NumPy arrays.

To run the script:
```bash
pip install scikit-learn tensorflow numpy
python3 src/preprocess.py
```

### 2.3. Baselines 
- We establish baselines to set a goal for the CNN-LSTM to beat
- Instead of a standard Bag-of-Words that looks at whole words, we use character n-grams (sequences of 2 to 5 characters). 
- isiZulu is agglutinative, a word-level TF-IDF would miss the root similarities between words like *ngihamba* and *uyahamba*. 
- Character n-grams naturally catch those shared morphological patterns (like *hamba*) and serve as a highly effective baseline. 
- Two models are trained to see which performs best:
   - Logistic Regression 
   - Support Vecotr Machine (LinearSVC) 
   