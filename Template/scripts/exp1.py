"""
Author: Zhou Chen
Date: 2020/2/2
Desc: desc
"""
import torch

# PyTorch版本
print(torch.__version__)
# 是否可用GPU
print(torch.cuda.is_available())
# 可用的GPU数目
print(torch.cuda.device_count())
# 可用的第一个GPU（默认从0编号开始）
print(torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "no gpu")