"""
Author: Zhou Chen
Date: 2020/3/21
Desc: desc
"""
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