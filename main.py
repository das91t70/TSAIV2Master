'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

# Training
def train(epoch):
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
        if args.scheduler == 'one_cycle_lr':
          scheduler.step()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
  parser.add_argument('--epochs', default=20, type=int, help='epochs')
  parser.add_argument('--resume', '-r', action='store_true',
                      help='resume from checkpoint')
  parser.add_argument('--optim', '-o', help='optimizer', default='adam')
  parser.add_argument('--scheduler', '-s', help='scheduler', default='one_cycle_lr')
  args = parser.parse_args()
  
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  best_acc = 0  # best test accuracy
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch
  
  # Data
  print('==> Preparing data..')
  transform_train = image_transforms(train=True)
  transform_test = image_transfroms(train=False)
  
  trainset = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=128, shuffle=True, num_workers=2)
  
  testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(
      testset, batch_size=100, shuffle=False, num_workers=2)
  
  classes = ('plane', 'car', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck')
  
  # Model
  print('==> Building model..')
  net = ResNet18()
  net = net.to(device)
  epochs = args.epochs
  
  if args.resume:
      # Load checkpoint.
      print('==> Resuming from checkpoint..')
      assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
      checkpoint = torch.load('./checkpoint/ckpt.pth')
      net.load_state_dict(checkpoint['net'])
      best_acc = checkpoint['acc']
      start_epoch = checkpoint['epoch']
  
  criterion = nn.CrossEntropyLoss()
  if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
  elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                        weight_decay=5e-4)
  
  if args.scheduler == 'cosine_annealing':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  elif args.scheduler == 'one_cycle_lr':
    lr_max = find_max_lr_rangetest("lsmith", model, train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=lr_max, epochs=epochs, steps_per_epoch=len(train_loader), final_div_factor=10, div_factor=10, pct_start=max_lr_epochs/EPOCHS, three_phase=False, anneal_strategy='linear')
  elif args.scheduler == 'reduced_lr_on_plateau':
    scheduler = ReduceLROnPlateau(optimizer, 'min')
  
  for epoch in range(epochs):
      train(epoch)
      test(epoch)
      if args.scheduler != 'onecyclelr':
        scheduler.step()
