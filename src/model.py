import torch
import torch.nn as nn
from itertools import cycle

from collections import OrderedDict


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=None, seed=42):
        super(QNetwork, self).__init__()
        if hidden_size is None:
            hidden_size = [128, 64, 32]
        self.seed = torch.manual_seed(seed)

        self.fc0 = nn.Linear(state_size, hidden_size[0])
        self.fc1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc2 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc3 = nn.Linear(hidden_size[2], action_size)

        layer_names = ["hl{}".format(counter) for counter in range(0, len(hidden_size) + 1)]
        layers = []
        layers += [(state_size, hidden_size[0])]
        layers += [(hidden_size[i-1], hidden_size[i]) for i in range(1, len(hidden_size))]
        layers += [(hidden_size[-1], action_size)]
        layers = [nn.Linear(layer[0], layer[1]) for layer in layers]
        layers_dict = OrderedDict(zip(layer_names, layers))

        relu_names = ["relu{}".format(counter) for counter in range(0, len(hidden_size))]
        relus = [nn.ReLU() for counter in range(0, len(hidden_size))]
        relu_dict = OrderedDict(zip(relu_names, relus))

        key_iterators = [iter(layers_dict.keys()), iter(relu_dict.keys())]
        values_iterators = [iter(layers_dict.values()), iter(relu_dict.values())]

        key_list = list(iterator.__next__() for iterator in cycle(key_iterators))
        value_list = list(iterator.__next__() for iterator in cycle(values_iterators))

        model_dict = OrderedDict(zip(key_list, value_list))

        self.model_sequential = nn.Sequential(model_dict)

    def forward(self, state):
        return self.model_sequential(state)
