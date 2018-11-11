import numpy as np
import tensorflow as tf
import pickle


def dec2bin(num, n_bits):
    return ("{0:0>%db}" % n_bits).format(num)


class AsciiSignalSource(object):
    """Class for constructing BFSK signals from ASCII symbols of a given text.
       Each bit frame is randomly subsampled based on sample rate and separation frequency

       see "Demodulation of Faded Wireless Signals using Deep Convolutional Neural Networks"
       http://sce2.umkc.edu/csee/beardc/DCNN%20Demodulation%20UMKC%20CCWC18.pdf
    """
    def __init__(self, path_to_text, frequency_mark=75e3, frequency_space=2*75e3, bit_rate=10**4, sample_rate=10 ** 6):
        """
        :param path_to_text: path to the text file used for signal generation
        :type path_to_text: string
        :param frequency_mark: frequency corresponding to logical "1" BFSK
        :type frequency_mark: float32
        :param frequency_space: frequency corresponding to logical "0" BFSK
        :type frequency_space: float32
        :param bit_rate: baud rate = bit rate
        :type  bit_rate: int
        :param sample_rate: sample rate
        :type sample_rate: int
        """
        self._frequency_mark = np.float32(frequency_mark)
        self._frequency_space = np.float32(frequency_space)
        self._bit_rate = bit_rate
        self._sample_rate = sample_rate
        self._samples_per_bit = self._sample_rate / self._bit_rate
        bit_time = np.linspace(0.0, 1.0 / self._bit_rate, int(self._samples_per_bit))
        self._frequencies = [np.sin(2 * np.pi * self._frequency_space * bit_time),
                             np.sin(2 * np.pi * self._frequency_mark * bit_time)]
        self._message = self._load_text(path_to_text)

    def _load_text(self, path_to_text):
        full_message = []
        for line in open(path_to_text):
            for letter in line.strip():
                full_message.append(ord(letter))
        return full_message

    def _generate_fsk_byte_frames(self, number):
        binary_form = dec2bin(number, 8)
        fsk_frame_sequence = []
        bit_sequence = []
        for digit in binary_form:
            digit = int(digit)
            bit_sequence.append(digit)
            fsk_frame_sequence.append(self._frequencies[digit])
        return fsk_frame_sequence, bit_sequence

    def generate_dataset(self):
        features = []
        labels = []
        for ascii_code in self._message:
            frame_sequence, bit_sequence = self._generate_fsk_byte_frames(ascii_code)
            for bit_frame, bit in zip(frame_sequence, bit_sequence):
                features.append(bit_frame)
                bit_label = np.zeros(2)
                bit_label[bit] = 1
                labels.append(bit_label)
        return np.array(features, dtype=np.float32), np.array(labels, np.int32)


class RealDataSource(object):
    """Class for reading real BFSK signals from CPKL
    """
    def __init__(self, source, exclude=None, include=None):
        """
        ;param source: cpkl-file with signal frames and corresponding labels
        """
        features = []
        labels = []
        with open(source, 'rb') as input_cpkl:
            dataset = pickle.load(input_cpkl)
        for i, f_list_and_l_list in enumerate(dataset):
            f_list = f_list_and_l_list[0]
            l_list = f_list_and_l_list[1]
            if include is None or i in include:
                if exclude is None or i not in exclude:
                    features.append(np.array(f_list))
                    labels.append(l_list)
        self._features = np.concatenate(features)
        flat_labels = np.concatenate(labels)
        self._labels = np.zeros([len(flat_labels), 2])
        for i, bit in enumerate(flat_labels):
            self._labels[i][bit] = 1

    def generate_dataset(self):
        return np.asarray(self._features, dtype=np.float32), np.array(self._labels, np.int32)


def add_noise_and_fft(signal, target_bit, snr_level=None):
    """
    Parsing function for tf.Dataset. Adds random noise of one of specified levels.
    :param signal: signal where noise will be added
    :type signal: tf.Tensor
    :param target_bit: bit value of the demodulated signal
    :type target_bit: tf.Tensor
    :param snr_level: target SNR levels for adding white noise
    :type snr_level: tf.placeholder scalar
    :return: pair (noisy_signal, target_bit)
    """
    if snr_level is not None:
        std = tf.pow(10.0, -snr_level/10)
        signal += tf.random_normal(shape=tf.shape(signal), mean=0.0, stddev=std, dtype=tf.float32)
    signal = tf.cast(signal, tf.complex64)
    power_fft = tf.abs(tf.fft(signal)[:signal.shape[0]//2])
    features = power_fft
    return features, target_bit

def add_noise(features, target_bit, snr_level=None):
    """
    Parsing function for tf.Dataset. Adds random noise of one of specified levels.
    :param signal: signal where noise will be added
    :type signal: tf.Tensor
    :param target_bit: bit value of the demodulated signal
    :type target_bit: tf.Tensor
    :param snr_level: target SNR levels for adding white noise
    :type snr_level: tf.placeholder scalar
    :return: pair (noisy_signal, target_bit)
    """
    if snr_level is not None:
        std = tf.pow(10.0, -snr_level/10)
        features += tf.square(tf.random_normal(shape=tf.shape(features), mean=0.0, stddev=std, dtype=tf.float32))
    return features, target_bit

