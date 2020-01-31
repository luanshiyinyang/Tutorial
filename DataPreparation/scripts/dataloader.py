"""
Author: Zhou Chen
Date: 2020/1/26
Desc: 通过DataLoader读取数据
"""
from my_dataset import MyDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


desc_train = '../data/desc_train.csv'
desc_valid = '../data/desc_valid.csv'
batch_size = 16
lr_init = 0.001
epochs = 10

normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(normMean, normStd)  # 按照imagenet标准
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(normMean, normStd)
])

# 构建MyDataset实例
train_data = MyDataset(desc_train, transform=train_transform)
valid_data = MyDataset(desc_valid, transform=valid_transform)

# 构建DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)


for epoch in range(epochs):
    for step, data in enumerate(train_loader):
        inputs, labels = data
        print("epoch", epoch, "step", step)
        print("data shape", inputs.shape, labels.shape)