# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:10:53 2023

@author: s.gorania
"""

import numpy as np
import torch


def preprocess_data(filename):

    # load ascii text and covert to lowercase
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X (a PyTorch tensor) to be [samples, time steps, features]
    X = torch.tensor(dataX,
                     dtype=torch.float32).reshape(n_patterns, seq_length, 1)
    # Normalise values in tensor X
    X = X / float(n_vocab)
    # Convert dataY into tensor y
    y = torch.tensor(dataY)

    # Print the shapes of X and y
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Return the processed data and relevant information
    return X, y, char_to_int, n_patterns, n_vocab
