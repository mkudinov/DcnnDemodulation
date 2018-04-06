import numpy as np


def dec2bin(num, n_bits):
    return ("{0:0>%db}" % n_bits).format(num)


class SignalGenerator(object):
    """Class for FSK signals generation with AWGN and Reileigh fading
    SNR can be specified for each signal."""

    def __init__(self, frequency_mark=984.0, frequency_space=966.0, baud_rate=2, sample_rate=14648):
        self.frequency_mark = frequency_mark
        self.frequency_space = frequency_space
        self.baud_rate = baud_rate
        self.sample_rate = sample_rate
        bit_time = np.linspace(0.0, 1.0 / self.baud_rate, int(self.sample_rate / self.baud_rate))
        self.mark_fragment = np.sin(2 * np.pi * self.frequency_mark * bit_time)
        self.space_fragment = np.sin(2 * np.pi * self.frequency_space * bit_time)

    def generate_fsk(self, number):
        binary_form = dec2bin(number, 16)
        # 1000 msec pause before signal start
        signal_fsk = np.zeros([self.sample_rate])
        for digit in binary_form:
            digit = int(digit)
            if digit == 1:
                signal_fsk = np.concatenate([signal_fsk, self.mark_fragment])
            else:
                signal_fsk = np.concatenate([signal_fsk, self.space_fragment])
        # 1000 msec pause after signal start
        signal_fsk = np.concatenate([signal_fsk, np.zeros([self.sample_rate])])
        return signal_fsk

    def generate_random_sample(self):
        pass
