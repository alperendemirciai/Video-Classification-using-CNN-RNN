## this archtecture was inspired by the following paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9190996 

import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionBlock(nn.Module):
    def __init__(self, time_steps):
        super(AttentionBlock, self).__init__()
        self.time_steps = time_steps
        self.dense = nn.Linear(time_steps, time_steps)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, inputs):
        # Permute (batch_size, time_steps, features) to (batch_size, features, time_steps)
        a = inputs.permute(0, 2, 1)
        
        # Dense layer (linear transformation)
        a = self.dense(a)
        
        # Softmax activation
        a = self.softmax(a)
        
        # Permute back to (batch_size, time_steps, features)
        a_probs = a.permute(0, 2, 1)
        output_attention_mul = inputs * a_probs
        
        return output_attention_mul


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attention = AttentionBlock(hidden_size * 2)  # Bi-directional -> hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, time_steps, input_size)

        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, time_steps, hidden_size*2)
        attention_out = self.attention(lstm_out)  # shape: (batch_size, time_steps, hidden_size*2)
        pooled_out = torch.mean(attention_out, dim=1)  # shape: (batch_size, hidden_size*2)

        # Final classification
        output = self.fc(pooled_out)  # shape: (batch_size, num_classes)

        return output
