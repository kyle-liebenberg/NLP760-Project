# Data 

## Related Files
- `src/build_author_dataset.py`: The srcipt to scrape Isolezwe for authors and their text samples
- `data/raw/isizulu_authors_dataset.csv`: The data outputted by the script

## Overview 
A custom-built dataset of 261 clean, distinct news articles scraped from the live Isolezwe news website. The dataset represents 11 distinct isiZulu authors.
The dataset is inherently small for deep learning standards. Texts vary in length and contain a mix of standard isiZulu, modern colloquialisms, and journalistic formatting.

## Purpose
It serves as the ground-truth corpus for training the authorship attribution models. It maps a specific text input to a specific author output.
Supervised machine learning requires labeled examples. By providing multiple distinct articles per author, the models can identify recurring stylometric (writing style) and morphological (word structure) patterns unique to each individual.
The data is structured in a tabular CSV format (`isizulu_authors_dataset.csv`) containing three primary features: the author's name, the source URL, and the raw text content of the article.

## Libraries used
- `pandas`: load, read, and manipulate this tabular data efficiently.