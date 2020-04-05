"""
Author: Zhou Chen
Date: 2020/4/5
Desc: desc
"""
from fastai.vision import data, learner, models
from fastai import metrics


if __name__ == '__main__':
    ds = data.ImageDataBunch.from_folder("../data/101_ObjectCategories/", valid_pct=0.2, size=128)
    learner_ = learner.cnn_learner(ds, models.resnet50, metrics=[metrics.accuracy, metrics.CrossEntropyFlat])
    learner_.fit(1)