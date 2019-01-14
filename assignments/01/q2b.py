# -*- coding: utf-8 -*-
__author__ = 'dongnanzhy'

""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
import sys

project_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_base, 'exampels'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time

from examples import utils

# Define paramaters for the model
learning_rate = 0.001
batch_size = 128
n_epochs = 50
n_train = 60000
n_test = 10000
num_hidden = 300

# Step 1: Read in data
mnist_folder = os.path.join(project_base, 'examples/data/mnist')
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)  # if you want to shuffle your data
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)


# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)  # initializer for test_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
_, num_features = img.get_shape()
_, num_logits = label.get_shape()

w1 = tf.get_variable('w1',
                    shape=[num_features, num_hidden],
                    initializer=tf.truncated_normal_initializer(mean=0,
                                                                stddev=0.1))
b1 = tf.get_variable('b1',
                    shape=[num_hidden],
                    initializer=tf.truncated_normal_initializer())
h1 = tf.tanh(tf.matmul(img, w1, name='matmul1') + b1,
             name='h1')

w2 = tf.get_variable('w2',
                    shape=[num_hidden, num_logits],
                    initializer=tf.truncated_normal_initializer(mean=0,
                                                                stddev=0.1))
b2 = tf.get_variable('b2',
                    shape=[num_logits],
                    initializer=tf.truncated_normal_initializer())

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = tf.add(tf.matmul(h1, w2, name='matmul2'),
                b2,
                name='add')


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,
                                               logits=logits,
                                               name='cross_entropy_loss'))


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   name='optimizer').minimize(loss,
                                                              name='train')


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits, name='softmax')
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)  # drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)  # drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds / n_test))
writer.close()
