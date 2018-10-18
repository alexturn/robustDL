import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torchvision
import torchvision.transforms as transforms

from net import MnistNet, MLP, VGG
from utils import simplex_grid

from tqdm import tqdm, trange

import numpy as np
from copy import deepcopy

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
# random
# testset.train_labels = torch.load('./random_labels_cifar.pth')
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss(reduction='sum')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda().long()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss

global_vals = []
for i in trange(3):
    # natural 
    weight_dict = torch.load('model_weights/vgg_weights_{}.pth'.format(i), 
                             map_location='cpu')
    # random                         
    # weight_dict = torch.load('model_weights/vgg_random_weights_{}.pth'.format(i), 
    #                          map_location='cpu')
    net = VGG('VGG16').cuda()
    net.load_state_dict(weight_dict)
    I_w = test(net, test_loader)

    vals = [] 
    for tick in trange(20):
        weight_dict_delta, delta = deepcopy(weight_dict),\
                                   deepcopy(weight_dict)
        
        norm = 0
        for key in list(weight_dict_delta.keys())[-2:]:
            delta[key] = torch.randn(delta[key].size())
            norm += delta[key].norm().pow(2)
        norm = norm.pow(0.5)

        I_w_delta, r = I_w, 0.
        while abs(I_w - I_w_delta) < 0.05:
            
            for key in list(weight_dict_delta.keys())[-2:]:
                weight_dict_delta[key] = weight_dict_delta[key].float() + delta[key] / norm

            net = VGG('VGG16').cuda()
            net.load_state_dict(weight_dict_delta)
            I_w_delta = test(net, test_loader)
            r += 1
            print(I_w_delta, I_w)
        
        vals += [r - 1]
    global_vals += [np.mean(vals)]

print(global_vals)
print("{} +/- {}".format(np.mean(global_vals), np.std(global_vals)))