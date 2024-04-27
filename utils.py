import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
import albumentations
import albumentations.pytorch
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def image_transforms(train):
  if train:
    return albumentations.Compose([
    albumentations.PadIfNeeded(min_height=36, min_width=36, p=1),
    albumentations.RandomCrop(size=(32,32), padding=4),
    albumentations.CoarseDropout(max_holes=1, max_height = 16, max_width=8, min_holes = 1, min_height=16, min_width=16, fill_value=(0.49139968, 0.48215827 ,0.44653124,), mask_fill_value = None, p=0.2),
    albumentations.augmentations.transforms.Normalize((0.49139968, 0.48215827 ,0.44653124,), (0.24703233,0.24348505,0.26158768,)),
        albumentations.pytorch.transforms.ToTensorV2()
    ])
  else:
    return albumentations.Compose([
        albumentations.augmentations.transforms.Normalize((0.49139968, 0.48215827 ,0.44653124,), (0.24703233,0.24348505,0.26158768,)),
        albumentations.pytorch.transforms.ToTensorV2()
        ])

# visualize dataloader in a batch
def visualize_data_in_batch(data_loader, start, end):
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    imshow(make_grid(images[start:end+1]), [classes[index] for index in labels.tolist()[start:end+1]])


# shpw images
def imshow(img, labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(labels)
    plt.show()

def find_max_lr_rangetest(test_type, model, train_loader, val_loader=None):
  if (test_type == "fastai"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-1)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
    _, max_lr = lr_finder.plot()
    lr_finder.reset()
  elif (test_type == "lsmith"):
    # leslie smith
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="linear")
    _, max_lr = lr_finder.plot(log_lr=False)
    lr_finder.reset()
  return max_lr


# transforms class - used for albumentations
class Transforms:
    def __init__(self, transforms: albumentations.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]
