import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example sequences of varying lengths
sequences = [
    [1, 2, 3],
    [4, 5, 6, 7, 8],
    [9, 10]
]

# Define the maximum sequence length
max_seq_length = 5

# Pad sequences to the same length
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')

print(padded_sequences)