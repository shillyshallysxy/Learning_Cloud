from Encoder_Decoder import text_helper, encoder_decoder
import tensorflow as tf
import os
import pickle
import numpy as np
from tqdm import tqdm

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

data_folder_name = '..\\temp'
data_path_name = 'cn_nlp\\translate'
vocab_name_cn = 'translate_cn.pkl'
vocab_name_en = 'translate_en.pkl'
train_record_name = 'translate_cn_en_train.tfrecord'
test_record_name = 'translate_cn_en_test.tfrecord'
test_string_cn = ['很高兴见到你！',
                  '你在这里干什么呢？',
                  '你的名字是什么？',
                  '中国越来越强大了。',
                  ]

vocab_path_cn = os.path.join(data_folder_name, data_path_name, vocab_name_cn)
vocab_path_en = os.path.join(data_folder_name, data_path_name, vocab_name_en)
model_save_path = os.path.join(data_folder_name, data_path_name, 'translate.ckpt')
model_log_path = os.path.join(data_folder_name, data_path_name, 'model_log.txt')

with open(vocab_path_cn, 'rb') as f:
    word_dict_cn = pickle.load(f)

with open(vocab_path_en, 'rb') as f:
    word_dict_en = pickle.load(f)


class Config(object):
    vocabulary_size_cn = len(word_dict_cn)
    vocabulary_size_en = len(word_dict_en)
    embedding_size_en = 200
    embedding_size_cn = 100
    retrain = True
    max_target_len = 200


class ConfigModel(object):
    vocabulary_size_cn = len(word_dict_cn)
    vocabulary_size_en = len(word_dict_en)
    embedding_size_en = 200
    embedding_size_cn = 100
    learning_rate = 0.005
    useTeacherForcing = True
    useBeamSearch = 1
    num_layers = 2
    hidden_size = 200


def main(argv):
    test_string_cn_num = text_helper.text_to_numbers(test_string_cn, word_dict_cn, 'cn')
    test_string_cn_len = [len(str_) for str_ in test_string_cn_num]
    with tf.Session() as sess:
        config_model = ConfigModel()
        config = Config()
        y_batch = [[0] * config.max_target_len]
        y_len = [config.max_target_len]

        encoder_embeddings = tf.Variable(tf.truncated_normal([config.vocabulary_size_cn,
                                                              config.embedding_size_cn], stddev=0.05),
                                         name='embeddings_cn')
        decoder_embeddings = tf.Variable(tf.truncated_normal([config.vocabulary_size_en,
                                                              config.embedding_size_en], stddev=0.05),
                                         name='embeddings_en')

        model = encoder_decoder.EncoderDecoderModel(word_dict_en, config_model, encoder_embeddings, decoder_embeddings)
        encoder_input = model.encoder_input
        encoder_len = model.encoder_length
        decoder_output = model.decoder_output
        decoder_len = model.decoder_length
        pred_out = model.out
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)

        for i, test_string_cn_num_every in enumerate(test_string_cn_num):
            feed_dict = {encoder_input: [np.array(test_string_cn_num_every)], encoder_len: [test_string_cn_len[i]],
                         decoder_output: y_batch, decoder_len: y_len}
            predict_out = sess.run(pred_out, feed_dict)
            print('Generation # {}. \n Out : {}'.
                  format(text_helper.numbers_to_text([test_string_cn_num_every[:test_string_cn_len[i]]], word_dict_cn),
                         text_helper.numbers_to_text(predict_out, word_dict_en),
                         ))


if __name__ == "__main__":
    tf.app.run()
