"""
Author: Zhou Chen
Date: 2020/3/24
Desc: desc
"""
from fastai.vision import data
import matplotlib.pyplot as plt

data = data.ImageDataBunch.from_folder("../data/101_ObjectCategories/", valid_pct=0.2, size=224)
train_ds, valid_ds = data.train_ds, data.valid_ds
img, label = train_ds[0]
img.show()
plt.show()

