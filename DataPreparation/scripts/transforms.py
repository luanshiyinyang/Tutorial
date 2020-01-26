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


# ---随机长宽比裁减---
tf = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=2)])


# ---上下左右中心裁减---
tf = transforms.Compose([transforms.FiveCrop(32)])


# ---上下左右中心裁减翻转---
tf = transforms.Compose([transforms.TenCrop(32)])


# ---概率水平翻转---
tf = transforms.Compose([transforms.RandomHorizontalFlip(0.9)])
rst = tf(img)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(rst)
plt.savefig('../assets/random_h_flip.png')

# ---概率垂直翻转---
tf = transforms.Compose([transforms.RandomVerticalFlip(0.9)])

# ---随机旋转---
tf = transforms.Compose([transforms.RandomRotation((30, 90), resample=False, expand=False, center=None)])
rst = tf(img)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(rst)
plt.savefig('../assets/random_rotation.png')

# ---图像尺寸调整---
tf = transforms.Compose([transforms.Resize(32, interpolation=2)])

# ---图像标准化---
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
tf = transforms.Compose([transforms.Normalize(normMean, normStd)])

# ---转为Tensor---
tf = transforms.Compose([transforms.ToTensor()])

# ---填充---
tf = transforms.Compose([transforms.Pad(None, fill=0, padding_mode='constant')])

# ---对比度等---
tf = transforms.Compose([transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)])

# ---灰度化---
tf = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

# ---线性变换---
tf = transforms.Compose([transforms.LinearTransformation(transformation_matrix=None)])

# ---仿射变换---
tf = transforms.Compose([transforms.RandomAffine(30, translate=None, scale=None, shear=None, resample=False, fillcolor=0)])

# ---概率灰度化---
tf = transforms.Compose([transforms.RandomGrayscale(0.5)])

# ---PIL---
tf = transforms.Compose([transforms.ToPILImage(mode=None)])

# ---lambda---
tf = transforms.Compose([transforms.Lambda(func)])

