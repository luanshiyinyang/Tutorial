"""
Author: Zhou Chen
Date: 2020/2/2
Desc: desc
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


model = Net()
num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
print(num_parameters)

params = list(model.named_parameters())
(name, param) = params[0]
print(name)
print(param.grad)

new_model = nn.Sequential()
for layer in model.named_modules():
    if isinstance(layer[1],nn.Conv2d):
         new_model.add_module(layer[0],layer[1])

model.load_state_dict(torch.load('model.pkl'), strict=False)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))