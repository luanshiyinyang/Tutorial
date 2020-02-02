"""
Author: Zhou Chen
Date: 2020/2/2
Desc: desc
"""
import torch
import torchvision
import numpy as np
from PIL import Image

# tensor转pil image
t = torch.randn(32, 3, 224, 224)
image = Image.fromarray(torch.clamp(t*255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(t)

# pil image转tensor
path = 'test.jpg'
tensor = torch.from_numpy(np.asarray(Image.open(path).convert('RGB'))).permute(2, 0, 1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(Image.open(path))