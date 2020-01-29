"""
Author: Zhou Chen
Date: 2020/1/29
Desc: 各种损失函数
"""
import torch
from torch import nn
pred = torch.ones(100, 1) * 0.5
label = torch.ones(100, 1)

l1_mean = nn.L1Loss()
l1_sum = nn.L1Loss(reduction='sum')

print(l1_mean(pred, label))
print(l1_sum(pred, label))

mse_loss = nn.MSELoss(reduction='mean')
print(mse_loss(pred, label))

ce_loss = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean')

kl = nn.KLDivLoss(reduction='mean')
print(kl(pred, label))

bce = nn.BCELoss(reduction='mean')

lbce = nn.BCEWithLogitsLoss()
lbce(pred, label)