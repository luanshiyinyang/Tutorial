"""
Author: Zhou Chen
Date: 2020/3/24
Desc: desc
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from dataset import Caltech101


def build_model(input_shape, n_classes):
    input_tensor = Input(shape=input_shape)
    vgg = keras.applications.VGG16(include_top=False, weights=None, input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(vgg.output)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = keras.models.Model(input_tensor, x)
    return model


model = build_model((224, 224, 3), 101)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(3e-4), metrics=['accuracy'])

callbacks = [
    keras.callbacks.TensorBoard("./logs")
]

train_gen, valid_gen, test_gen = Caltech101()

model.fit_generator(
    train_gen, steps_per_epoch=train_gen.n// train_gen.batch_size,
    epochs=50,
    callbacks=callbacks
)
