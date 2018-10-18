import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from net import MnistNet, MLP
from utils import simplex_grid

from tqdm import tqdm, trange

import numpy as np
from copy import deepcopy

weight_dict_1 = torch.load('model_weights/lenet_weights_0.pth', map_location='cpu')
weight_dict_2 = torch.load('model_weights/lenet_weights_1.pth', map_location='cpu')
weight_dict_3 = torch.load('model_weights/lenet_weights_2.pth', map_location='cpu')

x = np.linspace(-0.4, 1.3, 50)
y = np.linspace(-0.4, 1.3, 50)

X, Y = np.meshgrid(x, y)
Z = 1 - X - Y

grid = simplex_grid(3, 25) / 25
grid_val = []

def multiply_weights(state_dict, val):
    new_state_dict = deepcopy(state_dict)
    for key in new_state_dict.keys(): new_state_dict[key] *= val
    return new_state_dict

def sum_weights(list_of_state_dicts):
    for i in range(1, len(list_of_state_dicts)):
        for key in list_of_state_dicts[0].keys():
            list_of_state_dicts[0][key] += list_of_state_dicts[i][key]
    return list_of_state_dicts[0]

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=100, shuffle=True)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss

Z_ = []
for i in trange(X.shape[0]):
    Z_ += [[]]
    for j in trange(Y.shape[0]):
        convex_hull_weights = sum_weights([multiply_weights(weight_dict_1, X[i,j]),
                                        multiply_weights(weight_dict_2, Y[i,j]),
                                        multiply_weights(weight_dict_3, Z[i,j])
                                        ])

        net = MnistNet().cuda()
        net.load_state_dict(convex_hull_weights)
        Z_[i].append(test(net, test_loader))

np.save('./plots/X_mnist', X)
np.save('./plots/Y_mnist', Y)
np.save('./plots/Z_mnist', Z_)
