"""
Author: Zhou Chen
Date: 2020/1/28
Desc: 初始化方法
"""
import torch
from torch import nn
w = torch.randn((32, 3, 3))

torch.nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
torch.nn.init.xavier_normal_(w, gain=1)

nn.init.kaiming_uniform_(w, a=0, mode='fan_in', nonlinearity='leaky_relu')
nn.init.kaiming_normal_(w, a=0, mode='fan_in', nonlinearity='leaky_relu')