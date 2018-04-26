import numpy as np
import tensorflow as tf
import sys
import os

if __name__ == '__main__':
    path_to_signals = sys.argv[1]
    onlyfiles = {os.path.join(path_to_signals, f)[:-4] for f in os.listdir(path_to_signals) if os.path.isfile(os.path.join(path_to_signals, f))}
    for file in onlyfiles:
        binary_signal_file = file + '.dat'


