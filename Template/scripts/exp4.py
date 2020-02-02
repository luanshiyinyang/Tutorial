"""
Author: Zhou Chen
Date: 2020/2/2
Desc: desc
"""
import torch

t = torch.randn(2, 3, 4, 4)
# 张量的数据类型
print(t.type())
# 张量的维度信息
print(t.size())
# 张量的维度数量
print(t.dim())