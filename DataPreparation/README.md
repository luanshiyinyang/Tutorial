# Fastai数据准备


## 简介
数据是深度学习的立足之本，本文主要介绍Fastai框架如何进行数据加载与数据预处理。


## 模块划分
在之前的[Fastai简介文章](https://zhouchen.blog.csdn.net/article/details/89817650)我提到过，Fastai最核心的API是按照应用领域（任务类型）进行划分的，打开[官方文档](https://docs.fast.ai/applications.html)也会看到Fastai最核心的思路：**在一个`DataBunch`（Fastai的数据加载器）上训练一个`Model`对象，是非常简单的，只需要将数据和模型绑定到一个`Learner`对象即可。**

在Fastai的设计中，主要有四大应用领域，对应的四个模块名为`collab`（协同过滤问题）、`tabular`（表格或者结构化数据问题）、`text`（自然语言处理问题）以及`vision`（计算机视觉问题）。**本系列所有文章围绕都是图像数据进行处理，也就是说主要使用`vision`模块。****本系列所有文章围绕都是图像数据进行处理，也就是说主要使用`vision`模块。**

而在`vision`模块中


## 数据集构建
为了契合Fastai的API设计，这里并没有像之前Pytorch系列和Keras系列那样重构数据集为三个文件夹（对应训练集、验证集和测试集），这是考虑到Fastai的自动训练集划分的API的介绍，事实上划分数据集文件夹也是可以的，只不过多几个`DataBunch`对象而已。

关于数据集读取的API都在`fastai.vision.data`模块下，该模块定义了一个类`ImageDataBunch`用于处理大量的`Image`对象，这里的`Image`对象也是由`fastai.vision.data`模块下定义的。

在Fastai中数据集的处理都是基于`DataBunch`类的，`ImageDataBunch`是其子类，封装了很多具体的适合计算机视觉使用的方法