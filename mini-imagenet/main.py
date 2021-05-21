import argparse
import os
import random
import shutil
import time
import warnings
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
#import torch.optim
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from models.resnet_ws import l_resnet50, l_resnet18, l_resnet101

import torchvision.models as models
import math
import numpy as np
from torch.optim import lr_scheduler


import sys 
sys.path.append('../')
 
from myoptims.Diffgrad import diffgrad
from myoptims.tanangulargrad import tanangulargrad
from myoptims.cosangulargrad import cosangulargrad
from myoptims.AdaBelief import AdaBelief




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


def get_model(modelname):
    # create model
    num_classes=100
    if modelname=='r18':
        model = models.resnet18()
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    elif modelname=='r50':
        model = models.resnet50()
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif modelname=='r101':
        model = models.resnet101()
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif modelname=='r18ws':
      model = l_resnet18(num_classes=num_classes)
    elif modelname=='r50ws':
      model = l_resnet50(num_classes=num_classes)
    elif modelname=='r101ws':
      model = l_resnet101(num_classes=num_classes)
    else:
        print('==> Network not found...')
        exit()
    for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.uniform_()
                m.bias.data.zero_()
    return model


def get_loaders(args):
    print('==> Preparing MINI-Imagenet data...')
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
         ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,drop_last=True)


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader



def train(train_loader, model, criterion, optimizer, epoch, args):
    print('\nEpoch: %d' % epoch)
    model.train()
    total = 0
    train_loss = 0
    correct = 0
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.to('cuda'), target.to('cuda')

        output = model(input)
        loss = criterion(output, target)

        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

        train_loss += loss.item()
        total += target.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Training: Loss: {:.4f} | Acc: {:.4f}'.format(train_loss/(batch_idx+1),100.*correct/total))
    acc=100.*correct/total
    return acc, train_loss/(batch_idx+1)


def validate(val_loader, model, criterion, args):
    model.eval()

    val_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            val_loss +=loss.item()
        acc = 100.*correct/total
        print('Testing: Loss: {:.4f} | Acc: {:.4f}'.format(val_loss/(batch_idx+1), acc))
 
    return acc, val_loss/(batch_idx+1)


def main(args):
    args.arch = args.model
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    # Random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    model = get_model(args.model)
    if device == 'cuda':
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optim(args.alg, args.lr, model)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    
    train_loader, val_loader = get_loaders(args)


    best_acc = -1
    for epoch in range(args.start_epoch, args.epochs):
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        exp_lr_scheduler.step()
        val_acc, val_loss = validate(val_loader, model, criterion, args)

        if val_acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt' + '_' + args.model + '.t7')
            best_acc = val_acc
    print('Best Acc: {:.2f}'.format(best_acc))





if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Mini-ImageNet Training')

    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size')

    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')

    parser.add_argument('data', metavar='DIR', help='path to dataset')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--model', default='r50', type=str, help='model')
    parser.add_argument('--alg', default='adam', type=str, help='optimizer')
    args = parser.parse_args()
    main(args)
