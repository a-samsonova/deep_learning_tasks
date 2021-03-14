"""
Архитектура сети
"""
import torch
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self, layers_list):
        super(SimpleMLP, self).__init__()

        layers = []
        for t in layers_list:
            layers.append(nn.Linear(t[0], t[1]))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    layers_list_example = [(2, 3), (3, 4), (3, 3), (5, 5), (6, 2)]
    example_net = SimpleMLP(layers_list_example)
    print(example_net.net)
    x = torch.rand((1, 2))
    print(x)
    print(example_net.net(x))

    for i in example_net.parameters():
        print(i)

