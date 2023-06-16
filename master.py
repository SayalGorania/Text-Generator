# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:14:30 2023

@author: s.gorania
"""

from data_preprocessing import preprocess_data
from model_training_testing import CharModel, train_and_test_model

# Call the preprocess_data function
X, y, char_to_int, n_patterns, n_vocab = preprocess_data("wonderland.txt")


# Call the train_and_test_model function
best_model, best_loss = train_and_test_model(X,
                                             y,
                                             n_epochs=40,
                                             batch_size=128,
                                             n_vocab=n_vocab)

# Use the best_model or best_loss as needed for further steps
