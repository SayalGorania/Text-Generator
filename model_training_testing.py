# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:18:32 2023

@author: s.gorania
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class CharModel(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x


def train_and_test_model(X, y, n_epochs, batch_size, n_vocab, char_to_int):
    # Create DataLoader for training and validation
    loader = data.DataLoader(data.TensorDataset(X, y),
                             shuffle=True,
                             batch_size=batch_size)

    # Initialize the model
    model = CharModel(n_vocab)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    best_model = None
    best_loss = np.inf

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = model(X_batch)
                loss += loss_fn(y_pred, y_batch)

            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()

            print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))
    torch.save([best_model, char_to_int], "single-char2.pth")
    # Return the best trained model or any relevant evaluation results
    return best_model, best_loss
