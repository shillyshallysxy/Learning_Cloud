from Encoder_Decoder import text_helper, encoder_decoder
import tensorflow as tf
import os
import pickle
from tqdm import tqdm

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
model_save_path = os.path.join(data_folder_name, data_path_name, 'translate.ckpt')
cn_embeddings_save_path = os.path.join(data_folder_name, data_path_name, 'skipgram_embeddings_cn.ckpt')
en_embeddings_save_path = os.path.join(data_folder_name, data_path_name, 'skipgram_embeddings_en.ckpt')
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
    shuffle_pool_size = 1280
    batch_size = 64
    retrain = True


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


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text": tf.VarLenFeature(tf.int64),
                                                              "label": tf.VarLenFeature(tf.int64),
                                                              "text_length": tf.FixedLenFeature([], tf.int64),
                                                              "label_length": tf.FixedLenFeature([], tf.int64)})
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    text_ = tf.sparse_tensor_to_dense(features["text"])
    label_ = tf.sparse_tensor_to_dense(features["label"])
    text_lens_ = tf.cast(features["text_length"], tf.int32)
    label_lens_ = tf.cast(features["label_length"], tf.int32)
    return text_, label_, text_lens_, label_lens_


def get_dataset(record_name_):
    record_path_ = os.path.join(data_folder_name, data_path_name, record_name_)
    data_set_ = tf.data.TFRecordDataset(record_path_)
    return data_set_.map(__parse_function)


def main(argv):
    with tf.Session() as sess:
        config = Config()
        data_set_train = get_dataset(train_record_name)
        data_set_train = data_set_train.shuffle(config.shuffle_pool_size).repeat().\
            padded_batch(config.batch_size, padded_shapes=([None], [None], [], []))
        data_set_train_iter = data_set_train.make_one_shot_iterator()
        train_handle = sess.run(data_set_train_iter.string_handle())

        data_set_test = get_dataset(test_record_name)
        data_set_test = data_set_test.shuffle(config.shuffle_pool_size).repeat(). \
            padded_batch(config.batch_size, padded_shapes=([None], [None], [], []))
        data_set_test_iter = data_set_test.make_one_shot_iterator()
        test_handle = sess.run(data_set_test_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types,
                                                       data_set_train.output_shapes)
        text_, label_, text_length, label_length = iterator.get_next()
        # tokens_end = tf.ones([config.batch_size], dtype=tf.int64, name='tokens_eos') * word_dict_en['_EOS']
        # label_eos = tf.concat([label_[:, :], tf.reshape(tokens_end, [-1, 1])], 1)
        # label_length_eos = tf.add(label_length, 1)

        encoder_embeddings = tf.Variable(tf.truncated_normal([config.vocabulary_size_cn,
                                                              config.embedding_size_cn], stddev=0.05),
                                         name='embeddings_cn')
        decoder_embeddings = tf.Variable(tf.truncated_normal([config.vocabulary_size_en,
                                                              config.embedding_size_en], stddev=0.05),
                                         name='embeddings_en')

        config_model = ConfigModel()
        model = encoder_decoder.EncoderDecoderModel(word_dict_en, config_model, encoder_embeddings, decoder_embeddings)
        encoder_input = model.encoder_input
        decoder_output = model.decoder_output
        encoder_len = model.encoder_length
        decoder_len = model.decoder_length
        pred_out = model.out
        loss = model.loss
        train_op = model.train_op
        sess.run(tf.global_variables_initializer())

        if config.retrain:
            print('retraining')
            saver = tf.train.Saver()
            saver.restore(sess, model_save_path)
        else:
            saver_embedings_cn = tf.train.Saver({'embeddings_en': encoder_embeddings})
            saver_embedings_en = tf.train.Saver({'embeddings_en': decoder_embeddings})
            saver_embedings_cn.restore(sess, cn_embeddings_save_path)
            saver_embedings_en.restore(sess, en_embeddings_save_path)
        print('start training')
        saver = tf.train.Saver(max_to_keep=1)
        temp_train_loss = []
        temp_test_loss = []
        for i in range(500):
            x_batch, y_batch, x_batch_len, y_batch_len = sess.run([text_, label_, text_length, label_length],
                                                                  feed_dict={handle: train_handle})
            train_feed_dict = {encoder_input: x_batch, decoder_output: y_batch,
                               encoder_len: x_batch_len, decoder_len: y_batch_len}
            sess.run(train_op, train_feed_dict)
            # train_predict_out, train_loss = sess.run([pred_out, loss], train_feed_dict)
            train_predict_out, train_loss = sess.run([pred_out, loss], train_feed_dict)
            temp_train_loss.append(train_loss)
            if (i+1) % 100 == 0:
                test_x_batch, test_y_batch, test_x_batch_len, test_y_batch_len = \
                    sess.run([text_, label_, text_length, label_length], feed_dict={handle: test_handle})
                test_feed_dict = {encoder_input: test_x_batch, decoder_output: test_y_batch,
                                  encoder_len: test_x_batch_len, decoder_len: test_y_batch_len}
                test_predict_out, test_loss = sess.run([pred_out, loss], test_feed_dict)
                temp_test_loss.append(test_loss)
                print('Generation # {}. Train Loss : {:.3f} . '.format(i, train_loss))
                print('Generation # {}. Test Loss : {:.3f} . '.format(i, test_loss))
                print('Generation # {}. \n Target : {} . \n Out : {}'.
                      format(text_helper.numbers_to_text([test_x_batch[0][:test_x_batch_len[0]]], word_dict_cn),
                             text_helper.numbers_to_text([test_y_batch[0][:test_y_batch_len[0]]], word_dict_en),
                             text_helper.numbers_to_text([test_predict_out[0]], word_dict_en),
                             ))
                print('Generation # {}. \n Target : {} . \n Out : {}'.
                      format(text_helper.numbers_to_text([test_x_batch[1][:test_x_batch_len[1]]], word_dict_cn),
                             text_helper.numbers_to_text([test_y_batch[1][:test_y_batch_len[1]]], word_dict_en),
                             text_helper.numbers_to_text([test_predict_out[1]], word_dict_en),
                             ))

        saver.save(sess, model_save_path)
        # print(sess.run(model.decoder_embeddings))
        print('Generation . --model saved--')
        with open(model_log_path, 'w') as f:
            f.write('train_loss: ' + str(temp_train_loss))
            f.write('\n\ntest_loss: ' + str(temp_test_loss))
        print(' --log saved--')


if __name__ == '__main__':
    tf.app.run()
