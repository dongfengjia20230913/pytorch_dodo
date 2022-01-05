import torch
import torchvision
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_fashion_data(img_dir='./Datasets/FashionMNIST', batch_size = 256, num_workers = 4):
    mnist_train =torchvision.datasets.FashionMNIST(root=img_dir, train=True, download=True, transform=transforms.ToTensor())
    mnist_test =torchvision.datasets.FashionMNIST(root=img_dir, train=False, download=True, transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress','coat','sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()