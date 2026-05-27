"""
cnn_lstm_model.py
-----------------
Defines an advanced Concatenated CNN-LSTM hybrid architecture.
Combines structural temporal modeling with global stylistic feature bypass skips
to bridge the gap between traditional TF-IDF and deep models on small datasets.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, BatchNormalization, MaxPooling1D, LSTM, GlobalMaxPooling1D, concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_cnn_lstm(vocab_size: int, max_seq_length: int, num_classes: int, 
                   embedding_dim=128, filters=128, kernel_size=5, lstm_units=64, dropout_rate=0.4):
    """
    Builds a functional Keras model pairing an LSTM track with a Global CNN skip connection.
    """
    inputs = Input(shape=(max_seq_length,))
    
    # spatial dropout to prevent phrase memorization
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = SpatialDropout1D(0.3)(x)
    
    # convolutional feature extraction
    cnn_out = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
    cnn_out = BatchNormalization()(cnn_out)
    
    # temporal sequential pipeline
    pool_seq = MaxPooling1D(pool_size=2)(cnn_out)
    pool_seq = Dropout(dropout_rate)(pool_seq)
    lstm_out = LSTM(lstm_units, return_sequences=False)(pool_seq)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    global_pool = GlobalMaxPooling1D()(cnn_out)
    global_pool = Dropout(dropout_rate)(global_pool)
    
    merged = concatenate([lstm_out, global_pool])
    
    dense = Dense(64, activation='relu')(merged)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    
    outputs = Dense(num_classes, activation='softmax')(dense)
    
    # build functional model
    model = Model(inputs=inputs, outputs=outputs)
    
    # compile with Adam
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model