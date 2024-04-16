import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from normalization_utils import reverse_normalize

def train_model_with_lstm(features, targets, training_batch_size):
    # Define dimensions
    time_steps = features.shape[1]
    feature_count = features.shape[2]
    
    # Building the LSTM network
    network = Sequential([
        LSTM(128, activation='tanh', input_shape=(time_steps, feature_count), return_sequences=True, dropout=0.5),
        LSTM(128, activation='tanh', return_sequences=True, dropout=0.8, recurrent_dropout=0),
        Flatten(),
        Dense(100, activation='tanh'),
        Dropout(0.8),
        Dense(32, activation='tanh'),
        Dropout(0.8),
        Dense(1)
    ])
    
    # Compile the model
    network.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
    
    # Setup callbacks
    model_callbacks = [
        ModelCheckpoint("best_model.hdf5", monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    ]
    
    # Training the model
    training_history = network.fit(features, targets, epochs=10, 
                                   batch_size=training_batch_size, verbose=1, 
                                   validation_split=0.2, callbacks=model_callbacks, 
                                   shuffle=False)
    
    # Extract training and validation loss
    training_losses = training_history.history['loss']
    validation_losses = training_history.history['val_loss']
    
    return training_losses, validation_losses, network

def make_predictions(test_features, trained_model, normalizer):
    # Generate predictions
    raw_predictions = trained_model.predict(test_features)
    
    # Apply inverse transformation to predictions
    adjusted_predictions = reverse_normalize(raw_predictions, normalizer)
    
    return adjusted_predictions
