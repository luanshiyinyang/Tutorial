"""
Author: Zhou Chen
Date: 2020/4/4
Desc: desc
"""
from fastai.vision import transform, open_image, data
tfms = transform.get_transforms(do_flip=True)
data = data.ImageDataBunch.from_folder("../data/101_ObjectCategories/", valid_pct=0.2, size=224, ds_tfms=tfms)
train_ds, valid_ds = data.train_ds, data.valid_ds
