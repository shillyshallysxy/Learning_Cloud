import os
import jsonlines
import tensorflow as tf
from tqdm import tqdm
from SNLI_loader.text_helper import *
import pickle
from Transformer.transformer import *
data_folder_name = '..\\temp'
data_path_name = 'cn_nlp\\corpus\\SNLI\\snli_1.0'
test_snli_name = 'snli_1.0_test.jsonl'
test_snli_name_tf = 'snli_1.0_test.tfrecord'
train_snli_name = 'snli_1.0_train.jsonl'
train_snli_name_tf = 'snli_1.0_train.tfrecord'
vocab_name_en = 'translate_en_50.pkl'

cn_embeddings_save_path = os.path.join(data_folder_name, "cn_nlp\\translate", 'embeddings_cn.ckpt')
en_embeddings_save_path = os.path.join(data_folder_name, "cn_nlp\\translate", 'embeddings_en.ckpt')
model_save_path = os.path.join(data_folder_name, "cn_nlp\\snli")
model_name = "snli_model_{}"
transfer_model_save_path = os.path.join(data_folder_name, "cn_nlp\\snli", "snli_model_{}".format(1))
vocab_path_en = os.path.join(data_folder_name, "cn_nlp\\translate", vocab_name_en)

with open(vocab_path_en, 'rb') as f:
    word_dict_en = pickle.load(f)


class ConfigModel(object):
    vocab_size_en = len(word_dict_en)
    channels = 400
    learning_rate = 0.001
    layer_num = 6
    is_training = True
    is_transfer_learning = False
    restore_transfer_learning = False
    shuffle_pool_size = 2560
    dropout_rate = 0.1
    num_heads = 8
    batch_size = 64
    max_length = 100
    num_tags = 3


retrain_flag = True
test_total_acc = True
pm = ConfigModel()

data_path = os.path.join(data_folder_name, data_path_name)
label_to_num_dict = {'entailment': 0, "neutral": 1, 'contradiction': 2}


def get_data(snli_name, pad_=False):
    sentence_1 = list()
    sentence_2 = list()
    label = list()
    with open(os.path.join(data_path, snli_name), 'rb') as f:
        for item in jsonlines.Reader(f):
            if len(normalize_text(item["sentence1"])) > pm.max_length or len(normalize_text(item["sentence2"])) > pm.max_length:
                continue
            try:
                label.append(label_to_num_dict[item["gold_label"]])
            except KeyError:
                continue
            sentence_1.append(normalize_text(item["sentence1"]))
            sentence_2.append(normalize_text(item["sentence2"]))
    en_data_num_1 = text_to_numbers(sentence_1, word_dict_en)
    en_data_num_2 = text_to_numbers(sentence_2, word_dict_en)

    if pad_:
        for i in range(len(en_data_num_1)):
            en_data_num_1[i] = (en_data_num_1[i]+[0]*pm.max_length)[:pm.max_length]

            en_data_num_2[i] = (en_data_num_2[i]+[0]*pm.max_length)[:pm.max_length]
    return en_data_num_1, en_data_num_2, label


def write_binary(record_name, texts_, target_, label_):
    writer = tf.python_io.TFRecordWriter(record_name)
    for it, text in tqdm(enumerate(texts_)):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "text1": tf.train.Feature(int64_list=tf.train.Int64List(value=text)),
                    "text2": tf.train.Feature(int64_list=tf.train.Int64List(value=target_[it])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_[it]])),
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text1": tf.VarLenFeature(tf.int64),
                                                              "text2": tf.VarLenFeature(tf.int64),
                                                              "label": tf.FixedLenFeature([], tf.int64)})
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    text1_ = tf.sparse.to_dense(features["text1"])
    text2_ = tf.sparse.to_dense(features["text2"])
    label_ = tf.cast(features["label"], tf.int32)
    return text1_, text2_, label_


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
        # test_data1, test_data2, test_label = get_data(test_snli_name)
        # write_binary(os.path.join(data_path, test_snli_name_tf), test_data1, test_data2, test_label)
        # train_data1, train_data2, train_label = get_data(train_snli_name)
        # write_binary(os.path.join(data_path, train_snli_name_tf), train_data1, train_data2, train_label)
        # exit()
        # ===================================================================
        data_set_train = get_dataset(train_snli_name_tf)
        data_set_train = data_set_train.shuffle(pm.shuffle_pool_size).repeat(). \
            padded_batch(pm.batch_size, padded_shapes=([pm.max_length], [pm.max_length], []))
        data_set_train_iter = data_set_train.make_one_shot_iterator()
        train_handle = sess.run(data_set_train_iter.string_handle())

        data_set_test = get_dataset(os.path.join(test_snli_name_tf))
        if test_total_acc:
            data_set_test = data_set_test.shuffle(pm.shuffle_pool_size). \
                padded_batch(pm.batch_size, padded_shapes=([pm.max_length], [pm.max_length], []))
        else:
            data_set_test = data_set_test.shuffle(pm.shuffle_pool_size).repeat(). \
                padded_batch(pm.batch_size, padded_shapes=([pm.max_length], [pm.max_length], []))
        data_set_test_iter = data_set_test.make_one_shot_iterator()
        test_handle = sess.run(data_set_test_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types,
                                                       data_set_train.output_shapes)
        # cn     en        cn_len       en_len
        input1, input2, target = iterator.get_next()
        tsl = TransformerSNLICls(input1, input2, target, pm)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        # 是否载入之前的模型权重
        model_choose = "transfer_6"
        if retrain_flag:
            print("================retraining================")
            saver.restore(sess, os.path.join(model_save_path, model_name.format(model_choose)))
        if pm.restore_transfer_learning:
            print("loading transfer learning variables")
            var_list = tf.trainable_variables()
            layer_name_list = ["encoder_layer_" + str(i) for i in range(4)]
            var_list_ = [v for v in var_list if v.name.split("/")[1] in layer_name_list]
            var_list_ += [v for v in var_list if "lookup_table" in v.name]
            restore_saver = tf.train.Saver(var_list_)
            restore_saver.restore(sess, transfer_model_save_path)
        # 是否载入embedding
        if False:
            print("loading embeddings")
            graph = tf.get_default_graph()
            # a = sess.run(graph.get_tensor_by_name("encoder/en_embed/lookup_table:0"))
            embed_saver = tf.train.Saver({"embeddings_en": graph.get_tensor_by_name("encoder/en_embed/lookup_table:0")})
            embed_saver.restore(sess, en_embeddings_save_path)
            # b = sess.run(graph.get_tensor_by_name("encoder/en_embed/lookup_table:0"))
        if test_total_acc:
            try:
                total_acc = 0
                total_num = 0
                while True:
                    tpred, tacc, tloss = sess.run([tsl.preds, tsl.acc, tsl.loss], {handle: test_handle})
                    total_acc += tacc
                    total_num += 1
                    print("Generation test: acc: {}  loss: {} ".format(tacc, tloss))
            except tf.errors.OutOfRangeError:
                print(total_acc/total_num)
                exit()

        for i in range(8000):
            train_feed = {handle: train_handle}
            sess.run(target, train_feed)
        print("starting training")
        for i in range(4000):
            train_feed = {handle: train_handle}
            sess.run(tsl.train_op, train_feed)
            if (i+1) % 50 == 0:
                pred, acc, loss = sess.run([tsl.preds, tsl.acc, tsl.loss], train_feed)
                print("Generation train {} : acc: {}  loss: {} ".format(i, acc, loss))
            if (i+1) % 100 == 0:
                tpred, tacc, tloss = sess.run([tsl.preds, tsl.acc, tsl.loss], {handle: test_handle})
                print("Generation test {} : acc: {}  loss: {} ".format(i, tacc, tloss))
            if (i+1) % 1000 == 0:
                print("Generation train {} model saved ".format(i))
                saver.save(sess, os.path.join(model_save_path, model_name.format(model_choose)))
        saver.save(sess, os.path.join(model_save_path, model_name.format(model_choose)))









