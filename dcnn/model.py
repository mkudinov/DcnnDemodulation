import tensorflow as tf
import numpy as np


def variable_summaries(variable):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(variable))
        tf.summary.scalar('min', tf.reduce_min(variable))
        tf.summary.histogram('histogram', variable)


class DCNN:
    def __init__(self, input_placeholder, label_placeholder):
        self.input = tf.expand_dims(input_placeholder, -1)
        true_output = label_placeholder
        convolution_output = self.convlayers(self.input)
        self.logits = self.fc_layers(convolution_output)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_output, logits=self.logits))
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        tf.summary.scalar('cross_entropy', self.loss)
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(true_output, 1), tf.argmax(self.logits, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    @staticmethod
    def convlayers(input):
        # zero-mean input
        with tf.name_scope('preprocess'):
           normalized_input = input - tf.expand_dims(tf.reduce_mean(input, 2), 3)
           normalized_input = normalized_input / tf.expand_dims(tf.reduce_max(input, 2), 3)

        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 10, 1, 2], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(normalized_input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[2], dtype=tf.float32), trainable=True, name='biases')
            preactivate = tf.nn.bias_add(conv, biases)
            # tensorboard summary
            activation = tf.nn.relu(preactivate, name=scope)
            tf.summary.histogram('pre_activations', preactivate)
            tf.summary.histogram('activations', activation)
            variable_summaries(kernel)
            variable_summaries(biases)

        # conv2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 10, 2, 4], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4], dtype=tf.float32), trainable=True, name='biases')
            preactivate = tf.nn.bias_add(conv, biases)
            activation = tf.nn.relu(preactivate, name=scope)
            # tensorboard summary
            tf.summary.histogram('pre_activations', preactivate)
            tf.summary.histogram('activations', activation)
            variable_summaries(kernel)
            variable_summaries(biases)
        return activation

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
