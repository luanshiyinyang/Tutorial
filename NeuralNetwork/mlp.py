"""
Author: Zhou Chen
Date: 2019/10/10
Desc: 多层感知机
"""
import tensorflow as tf
x = tf.random.normal([2, 3])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2)
])
model.build(input_shape=[None, 3])
model.summary()
