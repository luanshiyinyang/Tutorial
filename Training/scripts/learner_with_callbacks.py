"""
Author: Zhou Chen
Date: 2020/4/5
Desc: desc
"""
from fastai.vision import data, learner, models
from fastai import metrics
from fastai import callbacks

ds = data.ImageDataBunch.from_folder("101_ObjectCategories/", valid_pct=0.2, size=128)
learner_ = learner.cnn_learner(ds, models.resnet50, metrics=[metrics.accuracy])
one_cycle = callbacks.OneCycleScheduler(learner_, lr_max=0.1)
learner_.fit(10, lr=3e-4, callbacks=[one_cycle, ])