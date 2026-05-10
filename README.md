# NLP760-Project
This project applies subword embeddings to authorship attribution, aiming to identify an author’s distinctive writing style by analysing subword patterns rather than whole words.


#Preprocessing
Huggingface tokenizer
from tokenizers import Tokenizer


# BPE Model
from tokenizers.models import BPE

# Pre-tokenization
from tokenizers.pre_tokenizers import Whitespace

