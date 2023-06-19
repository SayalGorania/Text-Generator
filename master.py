# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:14:30 2023

@author: s.gorania
"""

from data_preprocessing import preprocess_data
from model_training_testing import CharModel, train_and_test_model
import torch
from text_generation import generate_text

# Inputs
filename = "wonderland.txt"
seq_length = 500
n_epochs = 1
batch_size = 128
# Call the preprocess_data function
X, y, char_to_int, n_patterns, n_vocab = preprocess_data(filename)


# Call the train_and_test_model function
best_model, best_loss = train_and_test_model(X,
                                             y,
                                             n_epochs=n_epochs,
                                             batch_size=batch_size,
                                             n_vocab=n_vocab,
                                             char_to_int=char_to_int)

# Use the best_model or best_loss as needed for further steps


best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())

# Load the raw text and set the sequence length

raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# Call the generate_text function
generated_text = generate_text(best_model, char_to_int, int_to_char, n_vocab, seq_length, raw_text)

# Print the generated text
print(generated_text)
print("Done.")