import tensorflow as tf
import numpy as np
from dcnn.data_reader import AsciiSignalSource, add_noise_and_fft
from dcnn.model import DCNN
import os

flags = tf.flags
flags.DEFINE_string('data_dir', 'data',
                    'data directory. Should contain train_text..txt, valid_text.txt, test_text.txt')
flags.DEFINE_string('train_dir', 'cv', 'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('summaries_dir', 'summaries',
                    'directory to store tensorboard summaries')
flags.DEFINE_string('load_model', None,
                    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')
FLAGS = flags.FLAGS 

F1 = 984.0
F2 = 984.0 + 42
FS = 14648
BR = 2
BIT_LEN = int(FS / BR)


if __name__ == '__main__':
    SNR_levels = np.arange(0.0, 10.0, 2.0, np.float32)
    train_features, train_labels = AsciiSignalSource(os.path.join(FLAGS.data_dir, 'train_text.txt'), frequency_mark=F1,
                                                     frequency_space=F2, bit_rate=BR, sample_rate=FS).generate_dataset()
    feature_placeholder = tf.placeholder(train_features.dtype, train_features.shape)
    label_placeholder = tf.placeholder(train_labels.dtype, train_labels.shape)
    clean_training_set = tf.data.Dataset.from_tensor_slices((feature_placeholder, label_placeholder))
    training_set = clean_training_set.map(lambda x, y: add_noise_and_fft(x, y, SNR_levels[0]))
    for snr_level in SNR_levels[1:]:
        training_set.concatenate(clean_training_set.map(lambda x, y: add_noise_and_fft(x, y, snr_level)))
    training_set = training_set.shuffle(10000, reshuffle_each_iteration=True).batch(1)
    iterator = training_set.make_initializable_iterator()
    next_element = iterator.get_next()
    demodulation_cnn = DCNN(next_element[0], next_element[1])
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
        for epoch in range(10000):
            print("Epoch: %s" % epoch)
            sess.run(iterator.initializer,
                     feed_dict={feature_placeholder: train_features, label_placeholder: train_labels})
            while True:
                try:
                    _, summary_value = sess.run([demodulation_cnn.train_op, merged])
                    train_writer.add_summary(summary_value)
                except tf.errors.OutOfRangeError:
                    break
        print("Finished!")
