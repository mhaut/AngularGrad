'''Train CIFAR with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms


from torch.optim import lr_scheduler
import os
import argparse
from torchvision import datasets, models
from models import *


import sys 
sys.path.append('../')
 
from myoptims.Diffgrad import diffgrad
from myoptims.tanangulargrad import tanangulargrad
from myoptims.cosangulargrad import cosangulargrad
from myoptims.AdaBelief import AdaBelief

import random



def get_loaders(dsetname, bsize):
    print('==> Preparing ' + dsetname + ' data...')
    if dsetname == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        torchdset = torchvision.datasets.CIFAR10
    elif dsetname == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        torchdset = torchvision.datasets.CIFAR100
    else:
        print('==> Dataset not avaiable...')
        exit()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchdset(root='./data/'+dsetname+'/', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=4,drop_last=True)
    testset = torchdset(root='./data/'+dsetname+'/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    return trainloader, testloader


def get_model(modelname, Num_classes):
    if   modelname == 'v16':  net = VGG('VGG16',    Num_classes=Num_classes)
    elif modelname == 'r18':  net = ResNet18(       Num_classes=Num_classes)
    elif modelname == 'r34':  net = ResNet34(       Num_classes=Num_classes)
    elif modelname == 'r50':  net = ResNet50(       Num_classes=Num_classes)
    elif modelname == 'r101': net = ResNet101(      Num_classes=Num_classes)
    elif modelname == 'rx29': net = ResNeXt29_4x64d(Num_classes=Num_classes)
    elif modelname == 'dla':  net = DLA(            Num_classes=Num_classes)
    elif modelname == 'd121': net = DenseNet121(    Num_classes=Num_classes)
    else:
        print('==> Network not found...')
        exit()
    return net


def get_optim(optim_name, learning_rate, net):
    if   optim_name == 'sgd':            optimizer = optim.SGD(     net.parameters(), lr=learning_rate, momentum=0.9)
    elif optim_name == 'rmsprop':        optimizer = optim.RMSprop( net.parameters(), lr=learning_rate)
    elif optim_name == 'adam':           optimizer = optim.Adam(    net.parameters(), lr=learning_rate)
    elif optim_name == 'adamw':          optimizer = optim.AdamW(   net.parameters(), lr=learning_rate)
    elif optim_name == 'diffgrad':       optimizer = diffgrad(      net.parameters(), lr=learning_rate)
    elif optim_name == 'adabelief':      optimizer = AdaBelief(     net.parameters(), lr=learning_rate)
    elif optim_name == 'cosangulargrad': optimizer = cosangulargrad(net.parameters(), lr=learning_rate)
    elif optim_name == 'tanangulargrad': optimizer = tanangulargrad(net.parameters(), lr=learning_rate)
    else:
        print('==> Optimizer not found...')
        exit()
    return optimizer


def train(trainloader, epoch, net, optimizer, criterion, device='cuda'):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Training: Loss: {:.4f} | Acc: {:.4f}'.format(train_loss/(batch_idx+1),correct/total))
    acc=100.*correct/total
    return acc, train_loss/(batch_idx+1)


def test(testloader, epoch, net, criterion, device='cuda'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Testing:  Loss: {:.4f} | Acc: {:.4f}'.format(test_loss/(batch_idx+1),correct/total) )
    acc=100.*correct/total
    return acc, test_loss/(batch_idx+1)






def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.manualSeed)

    trainloader, testloader = get_loaders(args.dataset, args.bs)
    net = get_model(args.model, 10 if args.dataset == 'cifar10' else 100)

    if device == 'cuda':
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt' + '_' + args.dataset + '_' + args.model + '.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        best_acc = -1
        start_epoch = 0
        

    optimizer = get_optim(args.alg, args.lr, net)
    criterion = nn.CrossEntropyLoss()
    scheduler_lr = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

    
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_acc, train_loss = train(trainloader, epoch, net, optimizer, criterion, device=device)
        scheduler_lr.step()
        val_acc, val_loss = test(testloader, epoch, net, criterion, device=device)

        # Save checkpoint.
        if val_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt' + '_' + args.dataset + '_' + args.model + '.t7')
            best_acc = val_acc

    print('Best Acc: {:.2f}'.format(best_acc))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--dataset', type=str, default='cifar10', \
                                choices=['cifar10', 'cifar100'], \
                                help='dataset (options: cifar10, cifar100)')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--model', type=str, default='r50', \
                                choices=['v16', 'r18', 'r34', 'r50', 'r101', 'rx29', 'dla', 'd121'], \
                                help='dataset (options: v16, r18, r34, r50, r101, rx29, dla, d121)')
    parser.add_argument('--bs', default=128, type=int, help='batchsize')
    parser.add_argument('--alg', type=str, default='adam', \
                                choices=['sgd', 'rmsprop', 'adam', 'adamw', 'diffgrad', 'adabelief', 'cosangulargrad', 'tanangulargrad'], \
                                help='dataset (options: sgd, rmsprop, adam, adamw, diffgrad, adabelief, cosangulargrad, tanangulargrad)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--manualSeed', default=1111, type=int, help='random seed')

    args = parser.parse_args()

    main(args)

