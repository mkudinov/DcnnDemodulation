import numpy as np


def dec2bin(num, n_bits):
    return ("{0:0>%db}" % n_bits).format(num)


class SignalGeneratorASCII(object):
    """Class for constructing BFSK signals from ASCII symbols of a given text.
       Each bit frame is randomly subsampled based on sample rate and separation frequency

       see "Demodulation of Faded Wireless Signals using Deep Convolutional Neural Networks"
       http://sce2.umkc.edu/csee/beardc/DCNN%20Demodulation%20UMKC%20CCWC18.pdf
    """
    def __init__(self, path_to_text, frequency_mark=984.0, frequency_space=966.0, bit_rate=2, sample_rate=10 ** 6):
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
        self._frequency_mark = frequency_mark
        self._frequency_space = frequency_space
        self._bit_rate = bit_rate
        self._sample_rate = sample_rate
        bit_time = np.linspace(0.0, 1.0 / self._bit_rate, int(self._sample_rate / self._bit_rate))
        self._frequencies = [np.sin(2 * np.pi * self._frequency_space * bit_time),
                             np.sin(2 * np.pi * self._frequency_mark * bit_time)]
        self._message = self._load_text(path_to_text)
        self._samples_per_bit = self._bit_rate / self._sample_rate  # WARNING: MAGIC NUMBER FROM PAPER

    def _load_text(self, path_to_text):
        full_message = []
        for line in open(path_to_text):
            # 1000 msec pause before signal start
            full_message += [0] * self._bit_rate
            for letter in line.strip():
                full_message.append(ord(letter))
            # 1000 msec pause after signal start
            full_message += [0] * self._bit_rate
        return full_message

    def _generate_fsk_byte_frames(self, number):
        binary_form = dec2bin(number, 16)
        signal_fsk = []
        bit_sequence = []
        for digit in binary_form:
            digit = int(digit)
            bit_sequence.append(digit)
            signal_fsk.append(self._frequencies[digit])
        return signal_fsk, bit_sequence

    def __next__(self):
        for ascii_code in self._message:
            for bit_frame, bit in self._generate_fsk_byte_frames(ascii_code):
                yield bit, bit_frame


class AwgnParameterGenerator(object):
    """
    Helper class for adding noise with target SNR values
    """
    def __init__(self, snr_values):
        """
        :param snr_values: target SNR values for noise generation
        :type snr_values: list of float32
        """
        self.snr_values = snr_values

    def random_noise_parameter(self):
        """
        Get SNR for the next noisy signal frame
        :return: SNR value
        """
        return np.random.choice(self.snr_values, 1)[0]


class TfSignalSource(object):
    """
    Class is intended to transfer signal data in the form of Numpy arrays to tesorflow
    """
    def __init__(self, signal_generator, noise_generator):
        """
        :param signal_generator: signal generator
        :type signal_generator: SignalGeneratorASCII
        :param noise_generator: object for white noise generation
        :type noise_generator: AwgnGenerator
        """
        self._generator = signal_generator
