"""
Author: Zhou Chen
Date: 2020/1/29
Desc: 模型训练
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


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


net = Net()
x = torch.randn((32, 3, 224, 224))
y = torch.ones(32, ).long()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, dampening=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

epochs = 10
losses = []
for epoch in range(epochs):

    correct = 0.0
    total = 0.0

    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).squeeze().sum().numpy()
    losses.append(loss.item())
    print("loss", loss.item(), "acc", correct / total)

import matplotlib.pyplot as plt
plt.plot(list(range(len(losses))), losses)
plt.savefig('his.png')
plt.show()