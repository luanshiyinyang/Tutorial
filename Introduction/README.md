## MMDetection-简介

## 框架设计理念

在MMDetection中，Model（模型）被分为了如下5个模块：
1. **backbone**

    backbone是用于提取特征图的骨干网络，它是一个全卷积网络，例如ResNet、MobileNet等。
2. **neck**

    backbone和head之间的组件，通常为FPN等。
3. **head**

    具体的任务头，如边框预测或者类别预测。
4. **roi extractor**

    从特征图中提取RoI特征的组件，如RoI Align。
5. **loss**

    head部分计算损失的组件，训练必不可少的部分，常见的有FocalLoss、L1Loss等。
