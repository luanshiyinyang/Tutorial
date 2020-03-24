"""
Author: Zhou Chen
Date: 2020/3/23
Desc: desc
"""
import tensorflow.keras as keras


def Caltech101():

    train_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.,
        horizontal_flip=True,
        shear_range=0.2,
        width_shift_range=0.1
    )
    valid_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.
    )
    test_gen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.
    )

    batch_size = 32
    img_size = (224, 224)
    train_generator = train_gen.flow_from_directory("../data/Caltech101/train", batch_size=batch_size, target_size=img_size, class_mode='categorical')
    valid_generator = valid_gen.flow_from_directory('../data/Caltech101/valid', batch_size=batch_size, target_size=img_size, class_mode='categorical')
    test_generator = test_gen.flow_from_directory('../data/Caltech101/test', batch_size=batch_size, target_size=img_size, class_mode='categorical')
    return train_generator, valid_generator, test_generator