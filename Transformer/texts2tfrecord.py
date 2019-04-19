from Transformer import text_helper
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from nltk.corpus import stopwords
from tqdm import tqdm
import pickle
import os
ops.reset_default_graph()

sess = tf.Session()
create_tfrecord = True
data_folder_name = '..\\temp'
data_path_name = 'cn_nlp\\translate'
vocab_name_cn = 'translate_cn_50.pkl'
vocab_name_en = 'translate_en_50.pkl'
train_record_name = 'translate_cn_en_train_50.tfrecord'
test_record_name = 'translate_cn_en_test_50.tfrecord'

vocab_path_cn = os.path.join(data_folder_name, data_path_name, vocab_name_cn)
vocab_path_en = os.path.join(data_folder_name, data_path_name, vocab_name_en)

if create_tfrecord:
    # load movie review data
    en_data, en_len, cn_data, cn_len = text_helper.load_data()
    test_en_data, test_en_len, test_cn_data, test_cn_len = text_helper.load_test_data()

    if os.path.isfile(vocab_path_cn):
        with open(vocab_path_cn, 'rb') as f:
            word_dict_cn = pickle.load(f)
    else:
        print('creating dictionary')
        word_dict_cn = text_helper.build_dictionary(cn_data, 'cn')
        with open(vocab_path_cn, 'wb') as f:
            pickle.dump(word_dict_cn, f)

    if os.path.isfile(vocab_path_en):
        with open(vocab_path_en, 'rb') as f:
            word_dict_en = pickle.load(f)
    else:
        print('creating dictionary')
        word_dict_en = text_helper.build_dictionary(en_data, 'en')
        with open(vocab_path_en, 'wb') as f:
            pickle.dump(word_dict_en, f)

    vocabulary_size_cn = len(word_dict_cn)
    vocabulary_size_en = len(word_dict_en)
    en_max_len = max(en_len)
    cn_max_len = max(cn_len)
    print(vocabulary_size_cn, vocabulary_size_en)
    print(cn_max_len, en_max_len)
    # exit()
    cn_data_num = text_helper.text_to_numbers(cn_data, word_dict_cn, cn_len, 'cn')
    test_cn_data_num = text_helper.text_to_numbers(test_cn_data, word_dict_cn, test_cn_len, 'cn')
    en_data_num = text_helper.text_to_numbers(en_data, word_dict_en, en_len)
    test_en_data_num = text_helper.text_to_numbers(test_en_data, word_dict_en, test_en_len)
else:
    with open(vocab_path_en, 'rb') as f:
        word_dict_en = pickle.load(f)
    with open(vocab_path_cn, 'rb') as f:
        word_dict_cn = pickle.load(f)
    vocabulary_size_cn = len(word_dict_cn)
    vocabulary_size_en = len(word_dict_en)


def write_binary(record_name, texts_, target_, text_lens_, target_lens_):
    record_path = os.path.join(data_folder_name, data_path_name, record_name)
    writer = tf.python_io.TFRecordWriter(record_path)
    for it, text in tqdm(enumerate(texts_)):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text": tf.train.Feature(int64_list=tf.train.Int64List(value=text+[word_dict_cn['_EOS']])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=target_[it]+[word_dict_en['_EOS']])),
                    "text_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[text_lens_[it]+1])),
                    "label_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[target_lens_[it]+1]))
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


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


if create_tfrecord:
    print("creating tfrecord")
    write_binary(train_record_name, cn_data_num, en_data_num, cn_len, en_len)
    write_binary(test_record_name, test_cn_data_num, test_en_data_num, test_cn_len, test_en_len)
    # write_binary(train_record_name, en_data_num, cn_data_num, en_len, cn_len)
    # write_binary(test_record_name, test_en_data_num, test_cn_data_num, test_en_len, test_cn_len)
# exit()
record_path = os.path.join(data_folder_name, data_path_name, train_record_name)
dataset = tf.data.TFRecordDataset(record_path)
dataset = dataset.map(__parse_function)
data_train = dataset.shuffle(1000).repeat(10).padded_batch(1, padded_shapes=([None], [None], [], []))
iter_train = data_train.make_one_shot_iterator()
text_data, label_data,_,_ = iter_train.get_next()
with tf.Session() as sess:
    for i in range(10):
        print(sess.run([text_data, label_data]))
        print(sess.run(tf.shape(text_data)))
        # print(sess.run(iterator))
# print('')


handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, data_train.output_types, data_train.output_shapes)
x, y_, x_l_, y_l_ = iterator.get_next()

# table_name = 'text_table.txt'
# with open(table_name, 'w') as f:
#     for word in word_dict.values():
#         f.write(str(word)+'\n')
# text_lookup_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=table_name,
#                                                             num_oov_buckets=1)
# embed = tf.nn.embedding_lookup(embeddings, x)
# ids = text_lookup_table.lookup()
# sess.run(tf.tables_initializer())
handle_train = sess.run(iter_train.string_handle())
# print(handle_train)
# print(sess.run(ids, feed_dict={handle: handle_train}))
for i in range(1000):
    a, b, c, d = sess.run([x, y_, x_l_, y_l_], feed_dict={handle: handle_train})
    # print(a, b, c, d)
    if len(b[0]) != d[0] or len(a[0]) != c[0]:
        print(text_helper.numbers_to_text(b, word_dict_en), d[0])
        print(text_helper.numbers_to_text(a, word_dict_cn), c[0])

# print(sess.run(x_split, feed_dict={handle: handle_train}))

