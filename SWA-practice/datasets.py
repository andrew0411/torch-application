'''
CIFAR10 다운받아서 SWA로 학습해보기
'''

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_dataloader(args, valid_size=0.1, shuffle=True):

    train_normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    test_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        train_normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        test_normalize
    ])

    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    valid_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f'Train : {len(train_idx)}\nValid : {len(valid_idx)}\nTest : {len(test_dataset)}')

    return train_loader, valid_loader, test_loader

