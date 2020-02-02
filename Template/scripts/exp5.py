"""
Author: Zhou Chen
Date: 2020/2/2
Desc: desc
"""
import torch

# 设置默认数据类型
torch.set_default_tensor_type(torch.FloatTensor)

t = torch.randn(1)
print(t.type())
# 转为GPU数据类型
t = t.cuda()
# 转为CPU数据类型
t = t.cpu()
t = t.float()
t = t.long()