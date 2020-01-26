"""
Author: Zhou Chen
Date: 2020/1/26
Desc: 共22种transform方法
"""
# ---随机裁减---
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
img = Image.open('../data/test.jpg').convert('RGB')

tf = transforms.Compose([transforms.RandomCrop(32, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')])
rst = tf(img)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(rst)
plt.savefig('../assets/random_crop.png')


# ---中心裁减---
tf = transforms.Compose([transforms.CenterCrop(32)])
rst = tf(img)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(rst)
plt.savefig('../assets/center_crop.png')
plt.show()


