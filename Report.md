# Data set improvements 
### First data set:
- 161 articles 
- 7 authors 
- scraped from Isolzwe

### Second data set: 
- 263 articles 
- 11 authors 
- scraped from Isolzwe 
- Baselines perform well (84% SVM Baseline)
- CNN perform badly (classify all sample text to one author) 

### Third data set:
- 877 articles across 11 authors 
- Used selenium to click `load more` articles 
- Before using this, we want to modify the preprocessor to slice each article into 150 word chunks

---

# CNN Improvements:
### Version 1 
- Suffered due to low volume of data
- Results
```
F1 Macro  : 0.0101
Accuracy  : 0.0500
Precision : 0.0057
Recall    : 0.0455
```

### Version 2
- Expanded dataset and chunked it
- results
```
F1 Macro  : 0.1041
Accuracy  : 0.1732
Precision : 0.1092
Recall    : 0.1652
```

> What was wrong? 
- Under-capacity architecture: model was tweaked to only handle 184 rows of data (32 embedding dimensions, 32 filters, 16 LSTM units). Model was too small to learn and doesn't have the capacity to store what its learning. This leads to severe underfitting. 
- Untuned learning rate: loss suddenly jumped backwards on ecpoch 12. The optimiser is taking too large of steps, which causes it to overshoot the local minima and crash in to early stopping

### Version 3
- Tweaked the architecture and learning rates to adjust to the new volume of data 
- Results
```
F1 Macro  : 0.1358
Accuracy  : 0.2451
Precision : 0.2151
Recall    : 0.2224
```

>  What was wrong? 
- The embedding layer is starting from scratch. The TF-IDF baseline immediately receives global information about how unique subwords are across authors. The CNN-LSTM's `Embedding` layer starts with completely random weights. 
- The default learning rate in the Adam optimiser (0.0005) was pushing the weights around too violently for this size corpus, causing the gradients to explode or drift. 
- With standard architectures, adding deeper layers without normalizing activation scales between steps causes internal covariate shift. The CNN layer updates its features, completely throwing off the downstream LSTM layer.

### Version 4
- Batch Normalization: Placed after the 1D CNN. It rescales the layer outputs so the LSTM receives smooth, predictable inputs.
- Global Max Pooling: Swapping standard MaxPooling1D(pool_size=2) out for GlobalMaxPooling1D before passing data downstream. This acts as a primary stylistic keyword identifier, mirroring the behavior of the high-performing TF-IDF.
- A Learning Rate Callback (ReduceLROnPlateau): This tells TensorFlow: "If the validation loss stops improving for 2 epochs, automatically cut the learning rate in half so the optimizer can make microscopic, precise adjustments."
- Results: 
```
F1 Macro  : 0.3253
Accuracy  : 0.3529
Precision : 0.3398
Recall    : 0.3380
```

> What was wrong
- Overfitting: 
   - Training Accuracy: 78.29% 
   - Validation Accuracy: 38.56%
   - Model has plenty of parameters to capture patterns, but because the dataset is still relatively small for a half-million parameter network, the embedding and dense layers are starting to memorise the exact stylistic sentence fragments present in the 1428 chuncks. When it evaluates against completely separate validation chunks, it struggles to generalise, causing the validation loss to stall at 1.9124.
- There is also a subtle structural issue occurring inside the architecture: we used a standard MaxPooling1D layer. A standard max pooling layer downsamples spatial sequences locally. In text authorship attribution, an author's distinct stylistic signature (like a preferred subword combination or prefix arrangement) can appear anywhere in an article. Local pooling dilutes that signal.

### Version 5
- Introduce `GlobalMaxPooling1D` and a `RepeatVector` 
- Results: 
```
F1 Macro  : 0.0471
Accuracy  : 0.1013
Precision : 0.0503
Recall    : 0.1010
```

> What was wrong? 
- The Time Dimension Collapse: A standard Conv1D layer outputs a sequence of features (a matrix of shape [Batch, Time, Features]). The LSTM layer relies entirely on that Time dimension to find the chronological writing habits, sentence structure, and style flow of the author.
- The GlobalMaxPooling1D + RepeatVector Trap: When we introduced GlobalMaxPooling1D(), it collapsed the sequence down to a single static vector ([Batch, Features]), throwing away the sequence structure. By trying to fix it with RepeatVector(101), we duplicated that identical static vector 101 times. The downstream LSTM received an input sequence that was completely identical at every single timestamp. Because there was no sequential change over time, the LSTM's mathematical gates collapsed, rendering it unable to compute gradients.

### Version 6
- Removed `GlobalMaxPooling1D` bottleneck and restored true sequential feature transmission to the LSTM. 
- Results:
```
F1 Macro  : 0.0560
Accuracy  : 0.1405
Precision : 0.0758
Recall    : 0.1162
```

> What went wrong? 
- Severe Underfitting (High Bias)
- By cutting the model's feature extraction capacity in half (64 filters instead of 128) while simultaneously hitting it with heavy dropout and L2 weight decay, we choked the network's ability to learn. It simply did not have enough remaining mathematical capacity to find patterns in the text chunks, forcing it back into mode collapse (assigning a recall of 1.00 to Zimbili Vilakazi and 0.00 to almost everyone else).

### Version 7
- Restore CNN Filter Capacity: Bring the Conv1D filters back up to 128. This is essential because capturing the prefixes, roots, and suffixes of an agglutinative language like isiZulu requires a wider pool of spatial feature extractors.
- Loosen the Dropout Chokehold: Reduce the internal dropout rates to a moderate 0.3 (30%) during feature extraction. This leaves a cleaner signal for the network while still preventing it from memorizing specific training phrases.
- Remove Dense Head L2 Regularization: We will eliminate the L2 weight penalty on the dense head. The combination of structural dropout and your teammate's validation early stopping callback is already a sufficient defense against overfitting.
- Leverage Spatial Padding: We will maintain padding='same' on the convolutional boundary to preserve edge subword tokens (like paragraph-starting subject concords).