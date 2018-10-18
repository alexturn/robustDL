import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

mnist = torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

torch.manual_seed(42)
for i, label in enumerate(mnist.train_labels):
    if np.random.choice(a=[0,1],p=[0.5,0.5]) > 0:
        mnist.train_labels[i] = torch.randint(low=0, high=10, size=[1]).long().item()
    else:
        pass
labels_mnist = mnist.train_labels

torch.save(labels_mnist, './random_labels_mnist.pth')

for i, label in enumerate(cifar.train_labels):
    if np.random.choice(a=[0,1],p=[0.5,0.5]) > 0:
        cifar.train_labels[i] = torch.randint(low=0, high=10, size=[1]).long().item()
    else:
        pass
labels_cifar = cifar.train_labels

torch.save(labels_cifar, './random_labels_cifar.pth')