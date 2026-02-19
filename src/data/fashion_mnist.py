# src/data/fashion_mnist.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(data_dir, batch_size, num_workers, pin_memory, train_val_split):
    """
    下载并处理FashionMNIST 数据集， 返回DataLoaders
    """
    # 1. 定义数据变换(Transform)
    # FashionMNIST (1,28,28)，需转为Tensor并且进行归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. 下载数据集
    # 训练集(用于拆分为 Train/Val)
    full_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    # 测试集
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # 3. 划分训练集和验证集
    train_dataset, val_dataset = random_split(full_train_dataset, train_val_split)
    
    # 4. 构建Dataloders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader