# TensorFlow2迁移学习


## 简介
迁移学习的思路是指在较大的数据集（如ImageNet）上训练模型，得到泛化能力较强的模型，将其应用到较小数据集上，继续训练微调参数，从而可以在其他任务上使用。


## 自定义数据集
实际上，对于大型数据集而言，不可能采用之前的方法加载数据集。之前是将整个数据集读入为一个张量，每次从中取出一个batch的子张量，由于深度学习数据集一般都比较大，将这样大的数据集读入内存和显存是不现实的，因此一般分批次io取出数据。

而一般的CV数据集都是图片，且图片的存储格式有固定的规则，常见的主要有两种。第一种，分类别存放，每个类别一个目录，目录下放该类别的所有图片，如果有子类别则类似上面递归存储。另一种方法，所有文件在一个目录下，用一个csv文件记录所有图片信息，每一行有文件的id（一般是文件的相对目录），文件对应图片类别，其他annotation信息。显然第二种更加合理，现在，很多开放数据集都是综合二者进行存储。

应对文件读取，设计了如下的几个函数（数据集采用第一种方式存储，但是为了加载方便，生成了第二种存储的包含文件路径和标签的csv文件）。可以看到，在TensorFlow2中，数据集的加载是非常灵活的，没有固定的格式，按需读入即可。

数据预处理对于数据集有时候是必要的。resize（图片大小修改）大多数时候是必要的；Data Augmention（数据增广）可以增加小数据集的数据量，缓解过拟合问题，常见的增广手段有旋转、翻转、裁减等；标准化对于像素数值的处理可以使得模型的拟合更加容易。

下面的代码包含自定义数据集加载的代码。
```python
"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 自定义数据集加载
"""
import os
import glob
import random
import csv
import tensorflow as tf


def load_csv(root, filename, name2label):
    # root:数据集根目录
    # filename:csv文件名
    # name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            # 'pokemon\\mewtwo\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))

        # 1167, 'pokemon\\bulbasaur\\00000000.png'
        print(len(images), images)

        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                name = img.split(os.sep)[-2]
                label = name2label[name]
                # 'pokemon\\bulbasaur\\00000000.png', 0
                writer.writerow([img, label])
            print('written into csv file:', filename)

    # read from csv file
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 'pokemon\\bulbasaur\\00000000.png', 0
            img, label = row
            label = int(label)

            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)

    return images, labels


def load_pokemon(root, mode='train'):
    # 创建数字编码表
    name2label = {}  # "sq...":0
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())

    # 读取Label信息
    # [file1,file2,], [3,1]
    # 数据集全部数据
    images, labels = load_csv(root, 'images.csv', name2label)
    # 按照需求数据集类型去部分图片
    if mode == 'train':  # 60%
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':  # 20% = 60%->80%
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:  # 20% = 80%->100%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]

    return images, labels, name2label


# 使用imagenet的大样本均值和方差进行标准化
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def normalize(x, mean=img_mean, std=img_std):
    # x: [224, 224, 3]
    # mean: [224, 224, 3], std: [3]
    x = (x - mean) / std
    return x


def denormalize(x, mean=img_mean, std=img_std):
    """
    反标准化，用于数据可视化
    :param x:
    :param mean:
    :param std:
    :return:
    """
    x = x * std + mean
    return x


def preprocess(x, y):

    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)  # 指定3通道，取出alpha通道信息
    x = tf.image.resize(x, [244, 244])

    # data augmentation
    # x = tf.image.random_flip_up_down(x)  # 随机上下翻转，大多时候是无效的，本例无效
    x = tf.image.random_flip_left_right(x)  # 随机左右翻转，分类任务效果明显
    x = tf.image.random_crop(x, [224, 224, 3])

    # normalize
    # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1)
    x = normalize(x)

    y = tf.convert_to_tensor(y)

    return x, y


def main():
    import time

    images, labels, table = load_pokemon('pokemon', 'train')
    print('images', len(images), images)
    print('labels', len(labels), labels)
    print(table)

    # images: string path
    # labels: number
    db = tf.data.Dataset.from_tensor_slices((images, labels))
    db = db.shuffle(1000).map(preprocess).batch(32)

    writter = tf.summary.create_file_writer('logs')

    for step, (x, y) in enumerate(db):
        # x: [32, 224, 224, 3]
        # y: [32]
        with writter.as_default():
            x = denormalize(x)
            tf.summary.image('img', x, step=step, max_outputs=9)
            time.sleep(5)


if __name__ == '__main__':
    main()

```


## 迁移学习实战
在使用ResNet训练小数据集，效果很差（具体代码见文末Github），这是因为ResNet表示能力很强，很少的数据难以训练模型。这有两种解决方法，第一种是采用较小的模型，第二种则是采用迁移学习方法。第一种局限性很大，迁移学习是最常用的手段。

迁移学习的原理是基于CNN下层学习到的特征是非常底层的纹理特征，同类型任务这些特征是类似的，只需要重新学习高层特征即可。经典的网络集预训练参数可以通过keras模块获取。

最终的训练代码如下，由于只有最后一层全连接层参与训练，收敛较快。
```python
"""
Author: Zhou Chen
Date: 2019/11/4
Desc: About
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping
from pokemon import load_pokemon, normalize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)  # RGBA
    x = tf.image.resize(x, [244, 244])

    # x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224, 224, 3])

    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=5)

    return x, y


batchsz = 128

# creat train db
images, labels, table = load_pokemon('pokemon', mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
# crate validation db
images2, labels2, table = load_pokemon('pokemon', mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# create test db
images3, labels3, table = load_pokemon('pokemon', mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)

net = keras.applications.VGG19(weights='imagenet', include_top=False,
                               pooling='max')
net.trainable = False
newnet = keras.Sequential([
    net,
    layers.Dense(5)
])
newnet.build(input_shape=(4, 224, 224, 3))
newnet.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=5
)

newnet.compile(optimizer=optimizers.Adam(lr=1e-3),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
newnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=100,
           callbacks=[early_stopping])
newnet.evaluate(db_test)

```


## 补充说明
- 本文介绍了加载自定义数据集和迁移学习在TensorFlow2中的实现，更详细的可以查看官方文档。
- 具体的代码同步至[我的Github仓库](https://github.com/luanshiyinyang/Tutorial/tree/TensorFlow2)欢迎star；博客同步至我的[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看其他文章。
- 如有疏漏，欢迎指正。