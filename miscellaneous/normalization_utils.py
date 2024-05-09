import pandas as pd
import numpy as np

def create_sequences_normalize(dataframe, sequence_length):
    print("Shape of data:", dataframe.shape)
    seq_start_index = 0
    seq_features_list = []
    seq_targets_list = []
    normalization_base_list = []  # Stores the base for reversing normalization later
    
    for idx in range(len(dataframe) - sequence_length):
        seq_features = dataframe.iloc[seq_start_index:seq_start_index + sequence_length, :]
        seq_targets = dataframe.iloc[seq_start_index + 1:seq_start_index + 1 + sequence_length, 0]  
        # Target is assumed to be the first column
        
        # Perform window normalization
        base_for_normalization = seq_targets.iloc[0]
        seq_features_normalized = seq_features / seq_features.iloc[0, :] - 1
        seq_targets_normalized = seq_targets / seq_targets.iloc[0] - 1
        
        normalization_base_list.append([base_for_normalization] * sequence_length)
        
        # Append the normalized sequences and their corresponding targets to the lists
        seq_features_list.append(seq_features_normalized.values)
        seq_targets_list.append(seq_targets_normalized.values)
        
        seq_start_index += 1
    
    # Convert lists to numpy arrays and return them
    return np.array(seq_features_list), np.array(seq_targets_list), np.array(normalization_base_list)

def reverse_normalize(predicted_targets, normalization_bases):
    # Reverse the normalization process to obtain the original scale of predictions
    # Handle 3-D arrays (e.g., output of LSTM with return_sequences=True)
    if predicted_targets.ndim == 3:
        restored_targets = (predicted_targets[:, :, 0] * normalization_bases[:, 0]) + normalization_bases
    # Handle 2-D arrays (e.g., predicted values)
    elif predicted_targets.ndim == 2:
        restored_targets = (predicted_targets[:, 0] * normalization_bases[:, 0]) + normalization_bases[:, 0]
    
    return restored_targets
