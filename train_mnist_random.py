import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from net import MnistNet, MLP

def train(model, train_loader, optimizer, epoch):
    model.train()
    loss_track = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda().long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        loss_track += F.nll_loss(output, target, reduce='sum').item()
    print("Average loss on train: {:.6f}".format(loss_track / len(train_loader)))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda().long()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():

    for seed in range(42,45):
        torch.manual_seed(seed)

        model = MLP().cuda()
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0)

        train_ds = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        train_ds.train_labels = torch.load('./random_labels_mnist.pth').long()
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=1000, shuffle=True)

        # 10 epoches 
        for epoch in range(1, 150 + 1):
            train(model, train_loader, optimizer, epoch)
            test(model, test_loader)
        
        torch.save(model.state_dict(), './model_weights/mlp_random_weights_{}.pth'.format(seed))
        model.load_state_dict(torch.load('./model_weights/mlp_random_weights_{}.pth'.format(seed)))

if __name__ == '__main__':
    main()