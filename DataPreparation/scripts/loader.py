"""
Author: Zhou Chen
Date: 2020/3/24
Desc: desc
"""
from fastai import vision

ds = vision.ImageDataBunch.from_folder("../data/101_ObjectCategories/")
print(type(ds))