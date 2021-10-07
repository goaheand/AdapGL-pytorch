import torch


def build_fc_layers(input_feature, hidden_fc, activation='ReLU'):
    layers = torch.nn.Sequential(torch.nn.Linear(input_feature, hidden_fc[0]))
    act_func = getattr(torch.nn, activation, torch.nn.ReLU)
    for i in range(len(hidden_fc) - 1):
        layers.add_module('{}_{}'.format(activation, i), act_func())
        layer = torch.nn.Linear(hidden_fc[i], hidden_fc[i + 1])
        layers.add_module('fc_{}'.format(i), layer)
    return layers


class FC(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FC, self).__init__()
        self.fc = build_fc_layers(input_size, hidden_size)

    def forward(self, x):
        return self.fc(x)