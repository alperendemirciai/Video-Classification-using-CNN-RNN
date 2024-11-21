## this code is retrieved from https://github.com/kdgutier/residual_lstm 
## its an implementation of residual lstm in pytorch, the original paper: https://arxiv.org/pdf/1701.03360 

import torch
import torch.nn as nn
from typing import Tuple, Optional

class ResLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ii = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_ic = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ii = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(1 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(1 * hidden_size))
        self.weight_ir = nn.Parameter(torch.randn(hidden_size, input_size))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)

        ifo_gates = (torch.mm(input, self.weight_ii.t()) + self.bias_ii +
                     torch.mm(hx, self.weight_ih.t()) + self.bias_ih +
                     torch.mm(cx, self.weight_ic.t()) + self.bias_ic)
        ingate, forgetgate, outgate = ifo_gates.chunk(3, 1)
        
        cellgate = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        ry = torch.tanh(cy)

        if self.input_size == self.hidden_size:
            hy = outgate * (ry + input)
        else:
            hy = outgate * (ry + torch.mm(input, self.weight_ir.t()))

        if self.dropout:
            hy = self.dropout(hy)

        return hy, (hy, cy)


class ResLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ResLSTMCell(input_size, hidden_size, dropout)

    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = input.unbind(0)  # Split the sequence into timesteps
        outputs = []
        for step_input in inputs:
            out, hidden = self.cell(step_input, hidden)
            outputs.append(out)
        outputs = torch.stack(outputs)  # Combine the outputs for all timesteps
        return outputs, hidden


class ResLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.):
        super(ResLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            ResLSTMLayer(input_size if i == 0 else hidden_size, hidden_size, dropout)
            for i in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = input.size(1)

        # Initialize hidden states if not provided
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=input.device)
            c = torch.zeros(batch_size, self.hidden_size, device=input.device)
            hidden = (h, c)

        # Pass through all layers
        for layer in self.layers:
            input, hidden = layer(input, hidden)
        ##return input, hidden
        output = self.fc(input[-1])
        return output, hidden

# Example Usage
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    num_layers = 2

    model = ResLSTM(input_size, hidden_size, num_layers, dropout=0.2)
    input_data = torch.randn(seq_len, batch_size, input_size)  # Sequence input

    # No hidden state provided; will initialize automatically
    output, hidden_state = model(input_data)

    print("Output shape:", output.shape)
    print("Hidden state shapes:", hidden_state[0].shape, hidden_state[1].shape)
