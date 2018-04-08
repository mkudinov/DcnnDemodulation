import numpy as np


def dec2bin(num, n_bits):
    return ("{0:0>%db}" % n_bits).format(num)


class SignalGeneratorASCII(object):
    """Class for constructing BFSK signals from ASCII symbols of a given text.
       Each bit frame is randomly subsampled based on sample rate and separation frequency

       see "Demodulation of Faded Wireless Signals using Deep Convolutional Neural Networks"
       http://sce2.umkc.edu/csee/beardc/DCNN%20Demodulation%20UMKC%20CCWC18.pdf
    """

    def __init__(self, path_to_text, frequency_mark=984.0, frequency_space=966.0, baud_rate=2, sample_rate=10**6):
        """
        :param path_to_text (string): path to the text file used for signal generation
        :param frequency_mark (float32): frequency corresponding to logical "1" BFSK
        :param frequency_space (float32): frequency corresponding to logical "0" BFSK
        :param baud_rate (int): baud rate
        :param sample_rate (int): sample rate
        """
        self.frequency_mark = frequency_mark
        self.frequency_space = frequency_space
        self.baud_rate = baud_rate
        self.sample_rate = sample_rate
        bit_time = np.linspace(0.0, 1.0 / self.baud_rate, int(self.sample_rate / self.baud_rate))
        self.frequencies = [np.sin(2 * np.pi * self.frequency_space * bit_time),
                            np.sin(2 * np.pi * self.frequency_mark * bit_time)]
        self.message = self._load_text(path_to_text)
        self.subsampling_factor = 100  # WARNING: MAGIC NUMBER FROM PAPER

    def _load_text(self, path_to_text):
        full_message = []
        for line in open(path_to_text):
            # 1000 msec pause before signal start
            full_message += [0] * self.baud_rate
            for letter in line.strip():
                full_message.append(ord(letter))
            # 1000 msec pause after signal start
            full_message += [0] * self.baud_rate
        return full_message

    def _generate_fsk_byte_frames(self, number):
        binary_form = dec2bin(number, 16)
        signal_fsk = []
        bit_sequence = []
        for digit in binary_form:
            digit = int(digit)
            bit_sequence.append(digit)
            signal_fsk.append(self.frequencies[digit])
        return signal_fsk, bit_sequence

    def _subsample(self, bit_frame):
        return np.random.choice(bit_frame, self.subsampling_factor)

    def __next__(self):
        for ascii_code in self.message:
            for bit_frame, bit in self._generate_fsk_byte_frames(ascii_code):
                yield bit, self._subsample(bit_frame)
