__author__ = 'm.kudinov'


########################################################################################
# Mikhail Kudinov, 2018                                                                #
# Demodulation of Faded Wireless Signals using Deep Convolutional Neural Networks      #
# Details:                                                                             #
# http://sce2.umkc.edu/csee/beardc/DCNN%20Demodulation%20UMKC%20CCWC18.pdf             #
#                                                                                      #
########################################################################################

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    sess = tf.Session()
    vgg = DCNN()

    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]