"""
Author: Zhou Chen
Date: 2020/1/29
Desc: desc
"""
import torch
import torch.optim as optim


w1 = torch.randn(2, 2)
w2 = torch.randn(2, 2)

optimizer = optim.SGD([w1, w2], lr=0.1)
print(optimizer.param_groups)
optimizer.zero_grad()
print(optimizer.param_groups[0]['params'][0].grad)

sgd = optim.SGD(lr=0.001, momentum=0.9, weight_decay=0.001)

asgd= optim.Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)