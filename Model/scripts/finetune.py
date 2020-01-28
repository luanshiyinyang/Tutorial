"""
Author: Zhou Chen
Date: 2020/1/28
Desc: Finetune
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*54*54, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 101)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64*54*54)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = Net()
    # 假设训练完成了
    # 保存模型参数
    torch.save(net.state_dict(), 'net_weights.pkl')
    # 加载模型参数
    pretrained_params = torch.load('net_weights.pkl')
    # 构建模型，预训练参数初始化
    net = Net()
    net_state_dict = net.state_dict()
    pre_dict = {k: v for k, v in pretrained_params.items() if k in net_state_dict}
    net_state_dict.update(pre_dict)
