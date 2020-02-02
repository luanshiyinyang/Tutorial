"""
Author: Zhou Chen
Date: 2020/2/2
Desc: desc
"""
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'