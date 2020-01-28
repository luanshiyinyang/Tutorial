"""
Author: Zhou Chen
Date: 2020/1/28
Desc: 实现ResNet34
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    残差模块
    """

    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    def __init__(self, num_classes=101):
        super(ResNet34, self).__init__()
        self.model_name = 'ResNet34'
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        self.layer1 = self.make_layer(64, 128, 3)
        self.layer2 = self.make_layer(128, 256, 4, stride=2)
        self.layer3 = self.make_layer(256, 512, 6, stride=2)
        self.layer4 = self.make_layer(512, 512, 3, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels))

        layers = list()
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == '__main__':
    resnet = ResNet34(num_classes=101)
    data = torch.randn((32, 3, 224, 224))
    label = torch.zeros((32,))
    pred = resnet(data)
    print(pred.shape)