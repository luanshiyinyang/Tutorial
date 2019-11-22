# TensorFlow2对抗生成网络
>What i can not create, i do not understand. 我不能创造的东西，我当然不能理解它。

## 简介
对抗生成网络（GAN）是时下非常热门的一种神经网络，它主要用于复现数据的分布（distribution，或者数据的表示（representation））。尽管数据的分布非常的复杂，但是依靠神经网络强大的学习能力，可以学习其中的表示。其中，最典型的技术就是图像生成。GAN的出现是神经网络技术发展极具突破的一个创新。


## 原理
GAN网络由两个部分组成，它们是生成器（Generator）和判别器（Discriminator）。将输入数据与生成器产生的数据同时交给判别器检验，如果两者的分布接近，则表示生成器逐渐学习数据的分布，当接近到一定程度（判别器无法判别生成数据的真假），认为学习成功。


## 补充说明
- 本文介绍了GAN在TensorFlow2中的实现，更详细的可以查看官方文档。
- 具体的代码同步至[我的Github仓库](https://github.com/luanshiyinyang/Tutorial/tree/TensorFlow2)欢迎star；博客同步至我的[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看其他文章。
- 如有疏漏，欢迎指正。