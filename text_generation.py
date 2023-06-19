# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:27:28 2023

@author: s.gorania
"""

import torch.nn as nn

import numpy as np
import torch

def generate_text(best_model, char_to_int, int_to_char, n_vocab, seq_length, raw_text):
    class CharModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.linear = nn.Linear(256, n_vocab)
        def forward(self, x):
            x, _ = self.lstm(x)
            # take only the last output
            x = x[:, -1, :]
            # produce output
            x = self.linear(self.dropout(x))
            return x
    model = CharModel()
    model.load_state_dict(best_model)
    
    start = np.random.randint(0, len(raw_text) - seq_length)
    prompt = raw_text[start:start + seq_length]
    pattern = [char_to_int[c] for c in prompt]

    model.eval()
    generated_text = ""

    with torch.no_grad():
        for _ in range(1000):
            x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
            x = torch.tensor(x, dtype=torch.float32)
            prediction = model(x)
            index = int(prediction.argmax())
            result = int_to_char[index]
            generated_text += result
            pattern.append(index)
            pattern = pattern[1:]

    return generated_text
