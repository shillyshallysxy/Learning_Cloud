from Transformer.transformer import Transformer
import tensorflow as tf
from  Transformer.text_helper import *
import os
import pickle

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

data_folder_name = '..\\temp'
data_path_name = 'cn_nlp\\translate'
vocab_name_cn = 'translate_cn.pkl'
vocab_name_en = 'translate_en.pkl'
train_record_name = 'translate_cn_en_train.tfrecord'
test_record_name = 'translate_cn_en_test.tfrecord'

vocab_path_cn = os.path.join(data_folder_name, data_path_name, vocab_name_cn)
vocab_path_en = os.path.join(data_folder_name, data_path_name, vocab_name_en)
model_save_path = os.path.join(data_folder_name, data_path_name)
model_name = "translate_model_{}"
cn_embeddings_save_path = os.path.join(data_folder_name, data_path_name, 'skipgram_embeddings_cn.ckpt')
en_embeddings_save_path = os.path.join(data_folder_name, data_path_name, 'skipgram_embeddings_en.ckpt')
model_log_path = os.path.join(data_folder_name, data_path_name)

with open(vocab_path_cn, 'rb') as f:
    word_dict_cn = pickle.load(f)

with open(vocab_path_en, 'rb') as f:
    word_dict_en = pickle.load(f)


class ConfigModel(object):
    vocab_size_en = len(word_dict_en)
    vocab_size_de = len(word_dict_cn)
    channels = 400
    learning_rate = 0.001
    layer_num = 6
    is_training = True
    shuffle_pool_size = 256
    dropout_rate = 0.1
    num_heads = 8
    batch_size = 64
    max_length = 50


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text": tf.VarLenFeature(tf.int64),
                                                              "label": tf.VarLenFeature(tf.int64),
                                                              "text_length": tf.FixedLenFeature([], tf.int64),
                                                              "label_length": tf.FixedLenFeature([], tf.int64)})
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    text_ = tf.sparse.to_dense(features["text"])
    label_ = tf.sparse.to_dense(features["label"])
    text_lens_ = tf.cast(features["text_length"], tf.int32)
    label_lens_ = tf.cast(features["label_length"], tf.int32)
    return text_, label_, text_lens_, label_lens_


def get_dataset(record_name_):
    record_path_ = os.path.join(data_folder_name, data_path_name, record_name_)
    data_set_ = tf.data.TFRecordDataset(record_path_)
    return data_set_.map(__parse_function)


def main(argv):
    with tf.Session() as sess:
        pm = ConfigModel()

        data_set_train = get_dataset(train_record_name)
        data_set_train = data_set_train.shuffle(pm.shuffle_pool_size).repeat(). \
            padded_batch(pm.batch_size, padded_shapes=([None], [None], [], []))
        data_set_train_iter = data_set_train.make_one_shot_iterator()
        train_handle = sess.run(data_set_train_iter.string_handle())

        data_set_test = get_dataset(test_record_name)
        data_set_test = data_set_test.shuffle(pm.shuffle_pool_size).repeat(). \
            padded_batch(pm.batch_size, padded_shapes=([None], [None], [], []))
        data_set_test_iter = data_set_test.make_one_shot_iterator()
        test_handle = sess.run(data_set_test_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types,
                                                       data_set_train.output_shapes)
        text_, label_, text_length, label_length = iterator.get_next()
        transf = Transformer(text_, label_, pm=pm)

        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())
        for i in range(1000):

            sess.run(transf.train_op, {handle: train_handle})
            if (i+1) % 10 == 0:
                train_predict_out, train_loss = sess.run([transf.preds, transf.loss], {handle: train_handle})
                test_x_batch, test_y_batch, test_x_batch_len, test_y_batch_len = \
                    sess.run([text_, label_, text_length, label_length], {handle: train_handle})
                print('Generation {} # {}. \n Target : {} . \n Out : {}'.
                      format(i, numbers_to_text([test_x_batch[0][:test_x_batch_len[0]]], word_dict_cn),
                             numbers_to_text([test_y_batch[0][:test_y_batch_len[0]]], word_dict_en),
                             numbers_to_text([train_predict_out[0]], word_dict_en),
                             ))
                print('Generation {} # {}. \n Target : {} . \n Out : {}'.
                      format(i, numbers_to_text([test_x_batch[1][:test_x_batch_len[1]]], word_dict_cn),
                             numbers_to_text([test_y_batch[1][:test_y_batch_len[1]]], word_dict_en),
                             numbers_to_text([train_predict_out[1]], word_dict_en),
                             ))
        saver.save(sess, os.path.join(model_save_path, model_name.format("0")))


if __name__ == "__main__":
    tf.app.run()
