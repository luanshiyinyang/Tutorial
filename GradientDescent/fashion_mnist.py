"""
Author: Zhou Chen
Date: 2019/10/15
Desc: About
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batch_size = 64
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batch_size)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(batch_size)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # [b, 784] => [b, 256]
    layers.Dense(128, activation=tf.nn.relu),  # [b, 256] => [b, 128]
    layers.Dense(64, activation=tf.nn.relu),  # [b, 128] => [b, 64]
    layers.Dense(32, activation=tf.nn.relu),  # [b, 64] => [b, 32]
    layers.Dense(10),  # [b, 32] => [b, 10]
])
model.build(input_shape=([None, 28*28]))
optimizer = optimizers.Adam(lr=1e-3)


def main():
    # forward
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, 28*28])
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onthot = tf.one_hot(y, depth=10)

                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onthot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onthot, logits, from_logits=True))
            grads = tape.gradient(loss_ce, model.trainable_variables)
            # backward
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, "loss:", float(loss_mse), float(loss_ce))
        # test
        total_correct, total_num = 0, 0
        for x, y in db_test:
            x = tf.reshape(x, [-1, 28*28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))
            total_correct += int(correct)
            total_num += int(x.shape[0])
        acc = total_correct / total_num
        print("acc", acc)


if __name__ == '__main__':
    main()