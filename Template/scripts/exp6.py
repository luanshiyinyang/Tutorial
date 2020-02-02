"""
Author: Zhou Chen
Date: 2020/2/2
Desc: desc
"""
import torch

t = torch.randn(1, 2, 3)
# Tensor转ndarray
t = t.cpu().numpy()
# ndarray转tensor
t = torch.from_numpy(t).float()