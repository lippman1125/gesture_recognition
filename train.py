from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torchvision.models import resnet50
from data.dataset import hand_dataset
from torch.autograd import Variable
import pytorch_warmup as warmup
import numpy as np

from vgg import vgg
import shutil

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='enable warmup learning rate')
parser.add_argument('--warmup-epochs', type=int, default=5,
                    help='warmup epochs')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='enable mixup')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='mixup alpha')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    hand_dataset('./data', 'train.txt',
                transform=transforms.Compose([
                    transforms.Pad(8),
                    transforms.RandomResizedCrop((64,64),scale=(0.5,1.0), ratio=(0.8,1.2)),
                    # transforms.Resize((32,32)),
                    transforms.ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2)),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    hand_dataset('./data', 'test.txt',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# default vgg config
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
if args.refine:
    checkpoint = torch.load(args.refine)
    cfg = checkpoint['cfg']
    model = vgg(dataset="handdata", cfg=cfg)
    model.load_state_dict(checkpoint['state_dict'])
else:
    model = vgg(dataset="handdata", cfg=cfg)
    # model = resnet50(num_classes = 5)

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        cfg = checkpoint['cfg']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(epoch, lr_sched, warmup_sched):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data), Variable(target)
        inputs, targets = data.to(device), target.to(device)
        if args.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, args.cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        optimizer.zero_grad()
        outputs = model(inputs)
        if args.mixup:
            loss = mixup_criterion(F.cross_entropy, outputs, targets_a, targets_b, lam)
        else:
            loss = F.cross_entropy(outputs, targets)
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        lr_sched.step(lr_sched.last_epoch + 1)
        if warmup_sched is not None:
            warmup_sched.dampen()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tlr:{:f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data, optimizer.param_groups[0]['lr']))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        data, target = data.to(device), target.to(device)
        output = model(data)
        # test_loss += F.cross_entropy(output, target, size_average=False).data # sum up batch loss
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best_" + filename.split("_")[-1])

if args.sr:
    filename = "checkpoint_sparsity.pth.tar"
elif args.refine:
    filename = "checkpoint_finetune.pth.tar"
else:
    filename = "checkpoint_baseline.pth.tar"

num_steps = len(train_loader) * args.epochs
print("total iterations = {}".format(num_steps))
lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
if args.warmup:
    warmup_period = args.warmup_epochs * len(train_loader)
    print(warmup_period)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)
else:
    warmup_scheduler = None
best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    # if epoch in [args.epochs*0.25, args.epochs*0.5, args.epochs*0.75]:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
    train(epoch, lr_scheduler, warmup_scheduler)
    prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'cfg': cfg,
    }, is_best, filename)

print("Best Accuarcy = {:4f}".format(best_prec1))