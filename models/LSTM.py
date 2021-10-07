import torch
from .FC import build_fc_layers


class LSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_lstm, hidden_fc, out_size, bi=False):
        super(LSTMNet, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_lstm, batch_first=True, bidirectional=bi)
        if bi:
            hidden_lstm *= 2
        if hidden_fc is not None and len(hidden_fc) > 0:
            self.fc = build_fc_layers(hidden_lstm, hidden_fc)
            hidden_size = hidden_fc[-1]
        else:
            self.fc = None
            hidden_size = hidden_lstm
        self.out_layer = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x):
        y, _ = self.lstm(x)
        y = torch.relu(y[:, -1, :])
        if self.fc is not None:
            y = self.fc(y)
        y = self.out_layer(y)
        return y


class LSTM(torch.nn.Module):
    def __init__(self, **kwargs):
        super(LSTM, self).__init__()
        self.input_size = kwargs.get('input_size', 2)
        self.hidden_lstm = kwargs.get('hidden_lstm', 512)
        self.hidden_fc = kwargs.get('hidden_fc', None)
        self.num_nodes = kwargs.get('num_nodes', -1)
        self.output_size = kwargs.get('step_num_out', -1)

        self.lstm_list = torch.nn.ModuleList(
            [LSTMNet(self.input_size * self.num_nodes, self.hidden_lstm, self.hidden_fc, self.num_nodes) 
             for _ in range(self.output_size)]
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        y = []

        for lstm_layer in self.lstm_list:
            y.append(lstm_layer(x))

        y = torch.stack(y).transpose(0, 1)
        return y
