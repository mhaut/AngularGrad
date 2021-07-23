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
import torch.optim
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.optim import lr_scheduler

import numpy as np

import sys 
sys.path.append('../')
 
from myoptims.Diffgrad import diffgrad
from myoptims.tanangulargrad import tanangulargrad
from myoptims.cosangulargrad import cosangulargrad



def get_model(modelname, out_size):
    if modelname == 'r50p':
      model = models.resnet50(pretrained=True)
      model.fc = nn.Linear(in_features=2048, out_features=out_size, bias=True)
    elif modelname == 'r50':
      model = models.resnet50()
      model.fc = nn.Linear(in_features=2048, out_features=out_size, bias=True)
    else:
        print('==> Network not found...')
        exit()
    return model


def get_loaders(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
                    transforms.Resize(512),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomCrop(448),
                    transforms.ToTensor(),
                    normalize,
                ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(512),
                    transforms.CenterCrop(448),
                    transforms.ToTensor(),
                    normalize,
                ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,drop_last=True)
    
    return train_loader, val_loader



def train(train_loader, model, criterion, optimizer_base, optimizer_new, epoch, args):
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
        optimizer_new.zero_grad()
        optimizer_base.zero_grad()
        loss.backward()
        optimizer_new.step()
        optimizer_base.step()
        
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
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    class_num={'cub':200,'cars':196,'fgvc':100}

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    model = get_model(args.model, class_num[args.dataset])

    model = torch.nn.DataParallel(model).cuda()
    if device == 'cuda':
        model = model.cuda()
        #model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()


    new_param_ids = set(map(id, model.module.fc.parameters()))
    base_params = [p for p in model.parameters() if id(p) not in new_param_ids]
    param_groups_base =[{'params': base_params, 'lr_mult': 0.1}]
    param_groups_new=[{'params': model.module.fc.parameters(), 'lr_mult': 1.0}]


    if args.alg=='sgd':
        optimizer_base = optim.SGD(param_groups_base, args.lr, momentum=0.9)
        optimizer_new  = optim.SGD(param_groups_new, args.lr, momentum=0.9)
    elif args.alg=='rmsprop':
        optimizer_base = optim.RMSprop(param_groups_base, args.lr)
        optimizer_new  = optim.RMSprop(param_groups_new, args.lr)
    elif args.alg=='adam':
        optimizer_base = optim.Adam(param_groups_base, args.lr)
        optimizer_new  = optim.Adam(param_groups_new, args.lr)
    elif args.alg=='adamw':
        optimizer_base = optim.AdamW(param_groups_base, args.lr)
        optimizer_new  = optim.AdamW(param_groups_new, args.lr)
    elif args.alg=='diffgrad':
        optimizer_base = diffgrad(param_groups_base, args.lr)
        optimizer_new  = diffgrad(param_groups_new, args.lr)
    elif args.alg=='cosangulargrad':
        optimizer_base = cosangulargrad(param_groups_base, args.lr)
        optimizer_new  = cosangulargrad(param_groups_new, args.lr)
    elif args.alg=='tanangulargrad':
        optimizer_base = tanangulargrad(param_groups_base, args.lr)
        optimizer_new  = tanangulargrad(param_groups_new, args.lr)
    else:
        print('==> Optimizer not found...')
        exit()
    exp_lr_scheduler_new = lr_scheduler.MultiStepLR(optimizer_new, milestones=[30,50], gamma=0.1)
    exp_lr_scheduler_base = lr_scheduler.MultiStepLR(optimizer_base, milestones=[30,50], gamma=0.1)
    

    train_loader, val_loader = get_loaders(args)

    best_acc = -1
    datass = np.ones((4,args.epochs)) * -1000.0
    for epoch in range(args.start_epoch, args.epochs):
        train_acc, train_loss=train(train_loader, model, criterion, optimizer_base, optimizer_new, epoch, args)
        exp_lr_scheduler_new.step()
        exp_lr_scheduler_base.step()
        val_acc, val_loss = validate(val_loader, model, criterion, args)

        if val_acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
                'best_acc': best_acc,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Fine-Grained Training')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--model', default='r50p', type=str, help='model')
    parser.add_argument('--path', default='test', type=str, help='model')
    parser.add_argument('--alg', default='adam', type=str, help='algorithm')
    parser.add_argument('--dataset', default='cub', type=str, help='model')
    args = parser.parse_args()
    main(args)
