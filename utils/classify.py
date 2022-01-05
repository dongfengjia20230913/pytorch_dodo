import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')
import torchvision
import torchvision.transforms as transforms
import numpy as np

class Classify:
    def __init__(self, image_size = 28*28, class_nums = 10):
        self.W  = torch.tensor(np.random.normal(0,0.01, (image_size, class_nums)), dtype = torch.float)
        self.b = torch.zeros(class_nums, dtype = torch.float)
        self.W.requires_grad_(requires_grad=True)
        self.b.requires_grad_(requires_grad=True)
        self.num_inputs = image_size

    def softmax(self, X):
        X_exp = X.exp()
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition

    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat.gather(1, y.view(-1, 1)))

    def sgd(self, params, lr, batch_size): 
        for param in params:
            param.data -= lr * param.grad / batch_size # 注意这⾥里里更更改

    def evaluate_accuracy(self, data_iter, net):
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def net(self, X):
        xw = torch.mm(X.view(-1, self.num_inputs), self.W) + self.b # y = X.W+B
        return self.softmax(xw)#输出分类概率值

    def train(self, net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in train_iter:
                y_hat = net(X)
                l = loss(y_hat, y).sum()
                # 梯度清零
                if optimizer is not None:
                    optimizer.zero_grad()
                elif params is not None and params[0].grad is not None:
                    for param in params:
                        param.grad.data.zero_()
                l.backward()
                if optimizer is None:
                    self.sgd(params, lr, batch_size)
                else:
                    optimizer.step() # “softmax回归的简洁实现”⼀一节将⽤用到
                train_l_sum += l.item()
                train_acc_sum += (y_hat.argmax(dim=1) ==y).sum().item()
                n += y.shape[0]
            test_acc = self.evaluate_accuracy(test_iter, net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n,test_acc))