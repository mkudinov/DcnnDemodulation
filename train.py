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
from dcnn.data_reader import *

if __name__ == '__main__':
    SNR_levels = np.arange(10.0, 20.0, 2.0, np.float32)
    signal_generator = SignalGeneratorASCII("train_text.txt")
    training_set = None
    for snr_level in SNR_levels:
        if training_set is not None:
            training_set.concatenate(
                tf.data.Dataset.from_generator(signal_generator, signal_generator.output_types).map(
                    lambda x, y: add_noise_and_fft(x, y, snr_level)))
        else:
            training_set = tf.data.Dataset.from_generator(signal_generator, signal_generator.output_types).map(
                lambda x, y: add_noise_and_fft(x, y, snr_level))
    training_set.shuffle(10000, reshuffle_each_iteration=True)
