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
    SNR_levels = np.arange(10.0, 20.0, 2.0, np.float32)
