import argparse
import os
import sys
import shutil
import time
import math
import pickle
import pdb
from glob import glob
from pprint import pprint

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

N_CLASS = 512

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def group_list(x, y, group_size):
    for i in range(0, len(x), group_size):
        yield x[i: i+group_size], y[i: i+group_size]


def train(dataset_name):
    print(dataset_name)
    dataset = pickle.load(open(dataset_name, 'rb'))
    targets = []
    for i in range(N_CLASS):
        l = dataset['clustered_pool'][i].shape[0]
        targets += [i for _ in range(l)]
    X = np.concatenate(dataset['clustered_pool'])
    Y = np.array(targets)
    # cls = nn.Linear(X.shape[1], N_CLASS).cuda()
    cls = nn.Sequential(
            nn.Linear(X.shape[1], 512),
            nn.ReLU(True),
            nn.Linear(512, N_CLASS),
            ).cuda()
    # define optimizer
    optimizer = torch.optim.Adam(cls.parameters(), 1e-3, weight_decay=5e-4)
    criteria = nn.CrossEntropyLoss()
    min_loss = float('inf')

    for epoch in range(50):
        print(epoch)

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)


        index = np.random.permutation(X.shape[0])
        x = X[index]
        y = Y[index]

        index = 0
        acces = AverageMeter()
        losses = AverageMeter()
        for (data, target) in group_list(x, y, 128):
            data = torch.from_numpy(data).cuda()
            target = torch.from_numpy(target).cuda()

            optimizer.zero_grad()
            output = cls(data)
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()

            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()

            acces.update(correct / data.size(0))
            losses.update(loss.item())

        print(f'Epoch :{epoch}\tLoss: {losses.val}({losses.avg})\t'
              f'Acc: {acces.val}({acces.avg})')
        if losses.avg < min_loss:
            # torch.save(cls.state_dict(), dataset_name + '_cls.pth')
            torch.save(cls, dataset_name + '_cls.pth')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # set random seeds (maybe pass as argument)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    path = sys.argv[1]
    datasets = glob(path + '/*.KMeans')
    pprint(datasets)

    for dataset in datasets:
        train(dataset)

main()

