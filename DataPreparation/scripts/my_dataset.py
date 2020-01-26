"""
Author: Zhou Chen
Date: 2020/1/26
Desc: 自定义数据集
"""
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, desc_file, transform=None):
        self.all_data = pd.read_csv(desc_file).values
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.all_data[index, 0], self.all_data[index, 1]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.all_data)


if __name__ == '__main__':
    ds_train = MyDataset('../data/desc_train.csv', None)

