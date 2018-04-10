import tensorflow as tf
import numpy as np


class DCNN:
    def __init__(self, input_placeholder, label_placeholder):
        self.input = tf.expand_dims(input_placeholder, -1)
        true_output = label_placeholder
        convolution_output = self.convlayers(self.input)
        self.logits = self.fc_layers(convolution_output)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_output, logits=self.logits)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    @staticmethod
    def convlayers(input):
        # zero-mean input
        with tf.name_scope('preprocess'):
            normalized_input = input - tf.reduce_mean(input, 1)
            normalized_input = normalized_input / tf.reduce_max(input, 1)

        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 10, 1, 2], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(normalized_input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(out, name=scope)

        # conv2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 10, 2, 4], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(out, name=scope)
        return conv2

    @staticmethod
    def fc_layers(conv):
        # fc1
        with tf.name_scope('fc1'):
            shape = int(np.prod(conv.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            conv2_flat = tf.reshape(conv, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(conv2_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1l)

        # fc2
        with tf.name_scope('fc2'):
            fc2w = tf.Variable(tf.truncated_normal([256, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
            fc2 = tf.nn.relu(fc2l)

        # fc3
        with tf.name_scope('fc3'):
            fc3w = tf.Variable(tf.truncated_normal([64, 8], dtype=tf.float32, stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[8], dtype=tf.float32), trainable=True, name='biases')
            fc3 = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)

        # fc4
        with tf.name_scope('fc3'):
            fc4w = tf.Variable(tf.truncated_normal([8, 2], dtype=tf.float32, stddev=1e-1), name='weights')
            fc4b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32), trainable=True, name='biases')
            logits = tf.nn.bias_add(tf.matmul(fc3, fc4w), fc4b)
        return logits
