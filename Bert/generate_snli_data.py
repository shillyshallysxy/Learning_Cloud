import os
import jsonlines
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pickle
from Bert.tokenization import *
data_folder_name = '..\\temp'
data_path_name = 'cn_nlp\\corpus\\SNLI\\snli_1.0'
test_snli_name = 'snli_1.0_test.jsonl'
test_snli_name_tf = 'snli_1.0_test_bert.tfrecord'
train_snli_name = 'snli_1.0_train.jsonl'
train_snli_name_tf = 'snli_1.0_train_bert.tfrecord'

model_save_path = os.path.join(data_folder_name, "cn_nlp\\snli")
model_name = "snli_model_{}"
vocab_path = "uncased_L-12_H-768_A-12/vocab.txt"

word_dict_en = load_vocab(vocab_path)


class ConfigModel(object):
    vocab_size_en = len(word_dict_en)
    channels = 400
    learning_rate = 0.001
    layer_num = 4
    is_training = True
    is_transfer_learning = False
    restore_transfer_learning = False
    restore_embedding = False
    shuffle_pool_size = 2560
    dropout_rate = 0.1
    num_heads = 8
    batch_size = 64
    max_length = 103
    num_tags = 3


model_choose = "0"
retrain_flag = True
test_total_acc = False
pm = ConfigModel()

data_path = os.path.join(data_folder_name, data_path_name)
label_to_num_dict = {'entailment': 0, "neutral": 1, 'contradiction': 2}


def get_data(snli_name, pad_=False):
    sentence = list()
    mask = list()
    seg = list()
    label = list()
    ft = FullTokenizer(vocab_path, True)
    with open(os.path.join(data_path, snli_name), 'rb') as f:
        for item in jsonlines.Reader(f):
            if len(ft.tokenize(item["sentence1"])) > pm.max_length or len(ft.tokenize(item["sentence2"])) > pm.max_length:
                continue
            try:
                label.append(label_to_num_dict[item["gold_label"]])
            except KeyError:
                continue
            s1 = [word_dict_en["[CLS]"]] + ft.convert_tokens_to_ids(ft.tokenize(item["sentence1"])) + [word_dict_en["[SEP]"]]
            s2 = ft.convert_tokens_to_ids(ft.tokenize(item["sentence2"])) + [word_dict_en["[SEP]"]]
            lens1 = len(s1)
            lens2 = len(s2)
            sentence.append(s1+s2)
            mask.append((lens1+lens2)*[1])
            seg.append([0]*lens1+[1]*lens2)

    return sentence, mask, seg, label


def write_binary(record_name, texts_, mask_, seg_, label_):
    writer = tf.python_io.TFRecordWriter(record_name)
    for it, text in tqdm(enumerate(texts_)):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text": tf.train.Feature(int64_list=tf.train.Int64List(value=text)),
                    "mask": tf.train.Feature(int64_list=tf.train.Int64List(value=mask_[it])),
                    "seg": tf.train.Feature(int64_list=tf.train.Int64List(value=seg_[it])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_[it]]))
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text": tf.VarLenFeature(tf.int64),
                                                              "mask": tf.VarLenFeature(tf.int64),
                                                              "seg": tf.VarLenFeature(tf.int64),
                                                              "label": tf.FixedLenFeature([], tf.int64)})
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    texts_ = tf.sparse.to_dense(features["text"])
    mask_ = tf.sparse.to_dense(features["mask"])
    seg_ = tf.sparse.to_dense(features["seg"])
    label_ = tf.cast(features["label"], tf.int32)
    return texts_, mask_, seg_, label_


def get_dataset(record_name_):
    record_path_ = os.path.join(data_folder_name, data_path_name, record_name_)
    data_set_ = tf.data.TFRecordDataset(record_path_)
    return data_set_.map(__parse_function)


def generate_data(d1_, d2_, lb_, batch_size_=pm.batch_size):
    rand_indices = np.random.choice(len(d1_), batch_size_, replace=False)
    data_1 = np.array(d1_)[rand_indices]
    data_2 = np.array(d2_)[rand_indices]
    data_label = np.array(lb_)[rand_indices]
    return data_1, data_2, data_label


if __name__ == '__main__':
    with tf.Session() as sess:
        # ===================================================================
        # sentence, mask, seg, label = get_data(test_snli_name)
        # write_binary(os.path.join(data_path, test_snli_name_tf), sentence, mask, seg, label)
        # sentence, mask, seg, label = get_data(train_snli_name)
        # write_binary(os.path.join(data_path, train_snli_name_tf), sentence, mask, seg, label)
        # exit()
        # ===================================================================
        data_set_train = get_dataset(train_snli_name_tf)
        data_set_train = data_set_train.shuffle(pm.shuffle_pool_size).repeat(). \
            padded_batch(pm.batch_size, padded_shapes=([pm.max_length], [pm.max_length], [pm.max_length], []))
        data_set_train_iter = data_set_train.make_one_shot_iterator()
        train_handle = sess.run(data_set_train_iter.string_handle())

        data_set_test = get_dataset(os.path.join(test_snli_name_tf))
        if test_total_acc:
            data_set_test = data_set_test.shuffle(pm.shuffle_pool_size). \
                padded_batch(pm.batch_size, padded_shapes=([pm.max_length], [pm.max_length], [pm.max_length], []))
        else:
            data_set_test = data_set_test.shuffle(pm.shuffle_pool_size).repeat(). \
                padded_batch(pm.batch_size, padded_shapes=([pm.max_length], [pm.max_length], [pm.max_length], []))
        data_set_test_iter = data_set_test.make_one_shot_iterator()
        test_handle = sess.run(data_set_test_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types,
                                                       data_set_train.output_shapes)
        # cn     en        cn_len       en_len
        input1, input2, seg, target = iterator.get_next()
        a, b = sess.run([input1, input2], {handle: train_handle})
        print(1)










