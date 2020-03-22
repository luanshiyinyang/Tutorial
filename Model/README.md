# Keras 模型


## 简介
[前一篇文章](https://zhouchen.blog.csdn.net/article/details/89813050)介绍了以Keras为核心的数据相关的操作，当有了数据之后，就应该将数据“喂”给模型，这是深度学习中相当重要也非常有趣的一部分，本文主要涉及Keras中关于模型的一系列操作。包括模型定义，参数初始化以及模型的保存和加载。

## 模型构建
在介绍模型定义之前首先介绍在较为古老的版本Keras构建模型的两种手段，**Sequential容器**和**Function API（函数式API）**。

### Sequential
`keras.Sequential`是一个包装张量运算的容器，这里的张量运算主要指封装完成的`keras.layers`或者继承自`keras.Model`的张量运算（前者如卷积操作，后者如resnet block模块）。

该容器接收一个列表作为参数构建模型，列表中就是上述所说的各种张量运算，会返回一个Sequential类型的对象，该类其实是`keras.Model`的子类，拥有模型类的全部功能。例如下面构建的一个简单卷积分类器，其源码和输出如下。
```python
import tensorflow.keras as keras

model = keras.Sequential([
    keras.layers.Input(batch_input_shape=(None, 224, 224, 3)),
    keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(1000)]
)
print(model.summary())
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 112, 112, 16)      448       
_________________________________________________________________
flatten (Flatten)            (None, 200704)            0         
_________________________________________________________________
dense (Dense)                (None, 1000)              200705000 
=================================================================
Total params: 200,705,448
Trainable params: 200,705,448
Non-trainable params: 0
_________________________________________________________________
```

### Function API
下面介绍另一种堆叠张量运算的方式---函数式API，相对于Sequential那种单方向堆叠的API，Function API适合构建大型模型的使用，其最基本的用法为`张量运算(张量)`来堆叠张量的运算变换，再通过`keras.Model`类封装最终的模型。例如下面代码所描述的示例以及其输出结果，可以看到，和上文Sequential法的输出是一致的。
```python
import tensorflow.keras as keras

inputs = keras.layers.Input(batch_input_shape=(None, 224, 224, 3))
x = keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(inputs)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1000)(x)
model = keras.Model(inputs=inputs, outputs=x)
print(model.summary())
```

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
conv2d (Conv2D)              (None, 112, 112, 16)      448       
_________________________________________________________________
flatten (Flatten)            (None, 200704)            0         
_________________________________________________________________
dense (Dense)                (None, 1000)              200705000 
=================================================================
Total params: 200,705,448
Trainable params: 200,705,448
Non-trainable params: 0
_________________________________________________________________
```

### Subclassing API
上述的两种方法是原生Keras提供的两种模型构建手段，各有适用场景，一般按需使用即可。随着Keras加入TensorFlow大家族，新版本的Keras提供了一种新的类似Pytorch构建模型的API称为Subclassing API，其主要思路为继承`keras.Model`然后定义前向传播运算。

在Keras中想要让后续的训练模块认可一个模型的关键要素如下（这里特别说明）：
- 自定义模型以类的形式存在，且该类继承自`tf.keras.Model`并在`__init__`方法中声明模型需要的组件；
- 实现`call`方法，用于定义模型的前向运算；
- 在必要情况下实现compute_output_shape计算模型输出大小。

通过上述三个要求，可以构建相当灵活复杂的自定义模型，代价则是更加容易出错，一般情况下，Function API可以满足大多数设计要求。**Subclassing API更加适合于配合动态图机制进行细微的张量操作。**

定义模型后需要实例化模型，在使用模型前需要通过`model.build(input_shape)`来为模型指定输入尺寸，这样模型才能计算张量大小变化。

下面是和上面两种方法一样的模型使用Subclassing API构建的结果。
```python
import tensorflow.keras as keras
import tensorflow as tf


class MyModel(keras.Model):
    def __init__(self, num_classes=1000):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = keras.layers.Conv2D(16, (3, 3), (2, 2), padding='same', activation='relu')
        self.flatten = keras.layers.Flatten()
        self.classifier = keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


model = MyModel(1000)
model.build(input_shape=(32, 224, 224, 3))
print(model.summary())


model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['loss'])
```


## 模型保存与加载
上一节已经使用一些常见的张量操作来构建模型如卷积操作，具体有哪些已经封装的张量操作可以在`keras.layers`中找到，这里不多赘述，他们的参数也比较简单易懂。对于构建好的模型（不论是上一节提到的哪一种方法），都可以通过`keras.Model`类封装的方法保存训练模型或者训练参数，一般情况下，为了节省开销 建议保存参数即可。

只要是`keras.Model`对象或者其子类对象均可以通过`model.save()`方法进行模型存储，`keras.models.load_model()`从文件中加载一个模型，存储的文件默认是TensorFlow SavedModel文件。不过，需要尤其注意，使用Subclassing API的需要使用TensorFlow模型保存方法。

下面重点介绍使用最多的参数保存，对模型调用`model.save_weights()`即可保存模型参数，而构建模型对象调用`model.load_weights()`即可加载文件中的模型参数，同样支持TF CheckPoint格式和HDF5文件格式，这里建议使用Keras支持较好的HDF5文件，关于TensorFlow的操作可以查看[我相关文章](https://blog.csdn.net/zhouchen1998/category_9370890.html)。

示例代码如下。
```python
# 定义模型
model = MyModel(1000)
model.build(input_shape=(32, 224, 224, 3))
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['loss'])

# 训练模型，略过

# 保存模型参数
model.save_weights('model.h5')
del model

# 加载模型参数
model = MyModel(1000)
model.build(input_shape=(32, 224, 224, 3))
model.load_weights('model.h5')
```

模型的保存一般用于模型部署或者迁移学习，这方面的内容我会在后文提到，感兴趣可以查看[我的专栏文章](https://blog.csdn.net/zhouchen1998/category_8906034.html)。


## ResNet实战
下面主要使用Keras构建ResNet网络，主要是残差模块的设计，具体代码如下，采用Function API。
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Input, ZeroPadding2D, AveragePooling2D, Dense
from tensorflow.keras.layers import add


def Conv2D_BN(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='relu', name=conv_name)(x)
    x = BatchNormalization(name=bn_name)(x)
    return x


def identity_block(input_tensor, filters, kernel_size, strides=(1, 1), is_conv_shortcuts=False):
    """
    :param input_tensor:
    :param filters:
    :param kernel_size:
    :param strides:
    :param is_conv_shortcuts: 直接连接或者投影连接
    :return:
    """
    x = Conv2D_BN(input_tensor, filters, kernel_size, strides=strides, padding='same')
    x = Conv2D_BN(x, filters, kernel_size, padding='same')
    if is_conv_shortcuts:
        shortcut = Conv2D_BN(input_tensor, filters, kernel_size, strides=strides, padding='same')
        x = add([x, shortcut])
    else:
        x = add([x, input_tensor])
    return x


def bottleneck_block(input_tensor, filters=(64, 64, 256), strides=(1, 1), is_conv_shortcuts=False):
    """
    :param input_tensor:
    :param filters:
    :param strides:
    :param is_conv_shortcuts: 直接连接或者投影连接
    :return:
    """
    filters_1, filters_2, filters_3 = filters
    x = Conv2D_BN(input_tensor, filters=filters_1, kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2D_BN(x, filters=filters_2, kernel_size=(3, 3))
    x = Conv2D_BN(x, filters=filters_3, kernel_size=(1, 1))
    if is_conv_shortcuts:
        short_cut = Conv2D_BN(input_tensor, filters=filters_3, kernel_size=(1, 1), strides=strides)
        x = add([x, short_cut])
    else:
        x = add([x, input_tensor])
    return x


def ResNet34(input_shape=(224, 224, 3), n_classes=1000):
    """
    :param input_shape:
    :param n_classes:
    :return:
    """

    input_layer = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(input_layer)
    # block1
    x = Conv2D_BN(x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    # block2
    x = identity_block(x, filters=64, kernel_size=(3, 3))
    x = identity_block(x, filters=64, kernel_size=(3, 3))
    x = identity_block(x, filters=64, kernel_size=(3, 3))
    # block3
    x = identity_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2), is_conv_shortcuts=True)
    x = identity_block(x, filters=128, kernel_size=(3, 3))
    x = identity_block(x, filters=128, kernel_size=(3, 3))
    x = identity_block(x, filters=128, kernel_size=(3, 3))
    # block4
    x = identity_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2), is_conv_shortcuts=True)
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    x = identity_block(x, filters=256, kernel_size=(3, 3))
    # block5
    x = identity_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2), is_conv_shortcuts=True)
    x = identity_block(x, filters=512, kernel_size=(3, 3))
    x = identity_block(x, filters=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


def ResNet50(input_shape=(224, 224, 3), n_classes=1000):
    """
    :param input_shape:
    :param n_classes:
    :return:
    """
    input_layer = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(input_layer)
    # block1
    x = Conv2D_BN(x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # block2
    x = bottleneck_block(x, filters=(64, 64, 256), strides=(1, 1), is_conv_shortcuts=True)
    x = bottleneck_block(x, filters=(64, 64, 256))
    x = bottleneck_block(x, filters=(64, 64, 256))
    # block3
    x = bottleneck_block(x, filters=(128, 128, 512), strides=(2, 2), is_conv_shortcuts=True)
    x = bottleneck_block(x, filters=(128, 128, 512))
    x = bottleneck_block(x, filters=(128, 128, 512))
    x = bottleneck_block(x, filters=(128, 128, 512))
    # block4
    x = bottleneck_block(x, filters=(256, 256, 1024), strides=(2, 2), is_conv_shortcuts=True)
    x = bottleneck_block(x, filters=(256, 256, 1024))
    x = bottleneck_block(x, filters=(256, 256, 1024))
    x = bottleneck_block(x, filters=(256, 256, 1024))
    x = bottleneck_block(x, filters=(256, 256, 1024))
    x = bottleneck_block(x, filters=(256, 256, 1024))
    # block5
    x = bottleneck_block(x, filters=(512, 512, 2048), strides=(2, 2), is_conv_shortcuts=True)
    x = bottleneck_block(x, filters=(512, 512, 2048))
    x = bottleneck_block(x, filters=(512, 512, 2048))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


if __name__ == '__main__':
    resnet34 = ResNet34((224, 224, 3), n_classes=101)
    resnet50 = ResNet50((224, 224, 3), n_classes=101)
    print(resnet34.summary())
    print(resnet50.summary())
```


## 补充说明
本文主要介绍了三种API模式在Keras中构建深度模型，也提到了模型参数的保存与加载。具体的代码可以在[我的Github](https://github.com/luanshiyinyang/Tutorial/tree/Keras)找到，欢迎star或者fork。