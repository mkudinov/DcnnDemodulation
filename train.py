import tensorflow as tf
import numpy as np
from dcnn.data_reader import AsciiSignalSource, RealDataSource, add_noise_and_fft
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
flags.DEFINE_integer('test_signal_id', None, 'Id of the signal to be used as a test set')
flags.DEFINE_integer('valid_signal_id', None, 'Id of the signal to be used as a validation set')
FLAGS = flags.FLAGS 

F1 = 984.0
F2 = 984.0 + 42
FS = 14648
BR = 2
BIT_LEN = int(FS / BR)


if __name__ == '__main__':
    SNR_levels = [None]
    #train_features, train_labels = RealDataSource(os.path.join(FLAGS.data_dir, 'signals_dataset.cpkl'), exclude=[FLAGS.valid_signal_id, FLAGS.test_signal_id]).generate_dataset()
    train_features, train_labels = RealDataSource(os.path.join(FLAGS.data_dir, 'signals_dataset.cpkl')).generate_dataset()
    valid_features, valid_labels = RealDataSource(os.path.join(FLAGS.data_dir, 'signals_dataset.cpkl'), include=[FLAGS.valid_signal_id]).generate_dataset()
    test_features, test_labels = RealDataSource(os.path.join(FLAGS.data_dir, 'signals_dataset.cpkl'), include=[FLAGS.test_signal_id]).generate_dataset()
    training_set = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_features), tf.data.Dataset.from_tensor_slices(train_labels))).map(lambda x, y: add_noise_and_fft(x, y, SNR_levels[0])).shuffle(10000, reshuffle_each_iteration=True).batch(32)
    validation_set = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(valid_features), tf.data.Dataset.from_tensor_slices(valid_labels))).map(lambda x, y: add_noise_and_fft(x, y, SNR_levels[0])).batch(32)
    iterator = tf.data.Iterator.from_structure(training_set.output_types, training_set.output_shapes)
    next_element = iterator.get_next()
    validation_init_op = iterator.make_initializer(validation_set)
    demodulation_cnn = DCNN(next_element[0], next_element[1])
    merged = tf.summary.merge_all()
    step = 0
    loss_per_epoch = tf.placeholder(dtype=tf.float32)
    acc_per_epoch = tf.placeholder(dtype=tf.float32)
    loss_summary_per_epoch = tf.summary.scalar('loss_per_epoch', loss_per_epoch)
    acc_summary_per_epoch = tf.summary.scalar('acc_per_epoch', acc_per_epoch)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/valid')
        for epoch in range(10000):
            sess.run(iterator.make_initializer(training_set))
            train_loss = 0
            train_correct_predictions = 0
            while True:
                try:
                    _, loss, correct_prediction, summary_value = sess.run([demodulation_cnn.train_op, demodulation_cnn.loss, demodulation_cnn.correct_prediction, merged])
                    step += 1
                    train_loss += loss
                    train_correct_predictions += correct_prediction
                    train_writer.add_summary(summary_value, step)
                except tf.errors.OutOfRangeError:
                    break
            sess.run(iterator.make_initializer(validation_set))
            valid_loss = 0
            valid_correct_predictions = 0
            while True:
                try:
                    loss, correct_prediction, summary_value = sess.run([demodulation_cnn.loss, demodulation_cnn.correct_prediction, merged])
                    valid_loss += loss
                    valid_correct_predictions += correct_prediction
                except tf.errors.OutOfRangeError:
                    break
            train_writer.add_summary(
                loss_summary_per_epoch.eval(feed_dict={loss_per_epoch: train_loss / train_features.shape[0]}),
                epoch)
            train_writer.add_summary(
                acc_summary_per_epoch.eval(feed_dict={acc_per_epoch: train_correct_predictions / train_features.shape[0]}),
                epoch)
            valid_writer.add_summary(
                loss_summary_per_epoch.eval(feed_dict={loss_per_epoch: valid_loss / valid_features.shape[0]}),
                epoch)
            valid_writer.add_summary(
                acc_summary_per_epoch.eval(feed_dict={acc_per_epoch: valid_correct_predictions / valid_features.shape[0]}),
                epoch)
            print("Epoch: {}. Step: {} Train Loss: {} Valid Loss: {} Train acc.: {} Valid acc.: {}".format(epoch, step, train_loss / train_features.shape[0],
                        valid_loss / valid_features.shape[0], train_correct_predictions / train_features.shape[0], valid_correct_predictions / valid_features.shape[0]))

        print("Finished!")
