import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist2 = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.5
epochs = 10
batch_size = 100
regul_weight = 0.5 # I make this up
sparsity = 0.05
sparsity_weight = 10 #

def KL(rho, rho_j):
    return rho * tf.log(rho / rho_j) + (1 - rho) * tf.log((1 - rho) / (1 - rho_j))

data = tf.placeholder(tf.float32, [None, 784])
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.random_normal([784, 2000], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([2000]), name='b1')

W2 = tf.Variable(tf.random_normal([2000, 784], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([784]), name='b2')

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.sigmoid(hidden_out)

activation = tf.nn.sigmoid(tf.add(tf.matmul(data, W1), b1))

rho = tf.reduce_sum(activation, 0)

y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))
regul = tf.reduce_sum(W1 ** 2) + tf.reduce_sum(W2 ** 2)
penalty = tf.reduce_sum(KL(sparsity, rho))
loss = cross_entropy  + regul_weight * regul + sparsity_weight * penalty

optimiser = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(loss)

init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   train, _ = mnist2.train.next_batch(batch_size=len(mnist.train.labels))
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x,_ = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, loss],
                            feed_dict={x: batch_x, y: batch_x, data: train})
            avg_cost += c / total_batch
            z = sess.run(W1)
            print(W1)
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
