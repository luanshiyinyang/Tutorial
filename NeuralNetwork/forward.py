"""
Author: Zhou Chen
Date: 2019/10/9
Desc: About
"""

import tensorflow as tf
from tensorflow.keras import datasets

(x_train, y_train), (x_valid, y_valid) = datasets.mnist.load_data()

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
x_valid = tf.convert_to_tensor(x_valid, dtype=tf.float32)
y_valid = tf.convert_to_tensor(y_valid, dtype=tf.int32)
print(x_train.shape, y_train.shape)
print("min value of X is {}, max value of X is {}".format(tf.reduce_min(x_train), tf.reduce_max(x_train)))
print("min value of y is {}, max value of y is {}".format(tf.reduce_min(y_train), tf.reduce_max(y_train)))

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# [b, 784] => [b, 256] => [b, 128] => [b, 10]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.01))  # 梯度爆炸的可以通过较好的参数初始化
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.01))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.01))
b3 = tf.Variable(tf.zeros([10]))
lr = 1e-3

for epoch in range(10):
    # h = x@w+b
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape:
            h1 = x@w1 + b1  # 自动broadcast
            h1 = tf.nn.relu(h1)
            # [b, 256]=>[b, 128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128]=>[b, 10]
            out = h2@w3 + b3

            # compute loss
            # out: [b, 10]
            # y: [10]
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # update params, must inplace
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print("epoch {} step {}, loss {}".format(epoch, step, float(loss)))
    # test/evaluation
    total_correct, total_num = 0, 0
    for step, (x, y) in enumerate(test_db):
        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28*28])
        # [b, 784] => [b, 256] => [b, 128] => [b, 10]
        h1 = tf.nn.relu(x@w1) + b1
        h2 = tf.nn.relu(h1@w2) + b2
        out = h2@w3 + b3

        prob = tf.nn.softmax(out, axis=-1)
        pred = tf.argmax(prob, axis=-1)
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)  # boolean to int
        correct = tf.reduce_sum(correct)

        total_correct += int(correct)
        total_num += x.shape[0]
    acc = total_correct / total_num
    print("validation accuracy", acc)



