import tensorflow as tf
from Bert.modeling import *
import os
from Bert.tokenization import *
from Bert.optimization import *
pathname = "uncased_L-12_H-768_A-12/bert_model.ckpt"  # 模型地址
bert_config = BertConfig.from_json_file("uncased_L-12_H-768_A-12/bert_config.json")  # 配置文件地址。
vocab_path = "uncased_L-12_H-768_A-12/vocab.txt"
data_folder_name = '..\\temp'
data_path_name = 'cn_nlp\\corpus\\SNLI\\snli_1.0'
test_snli_name_tf = 'snli_1.0_test.tfrecord'
train_snli_name_tf = 'snli_1.0_train.tfrecord'
model_save_path = os.path.join(data_folder_name, "cn_nlp\\snli")
model_name = "snli_model_{}"
model_choose = "bert"


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


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


configsession = tf.ConfigProto()
configsession.gpu_options.allow_growth = True
test_total_acc = False
config = copy.deepcopy(bert_config)
sess = tf.Session(config=configsession)


with sess.as_default():

    vocab_en = load_vocab(vocab_path)

    data_set_train = get_dataset(train_snli_name_tf)
    data_set_train = data_set_train.shuffle(config.shuffle_pool_size).repeat(). \
        padded_batch(config.batch_size, padded_shapes=([config.max_length], [config.max_length], [config.max_length], []))
    data_set_train_iter = data_set_train.make_one_shot_iterator()
    train_handle = sess.run(data_set_train_iter.string_handle())

    data_set_test = get_dataset(os.path.join(test_snli_name_tf))
    if test_total_acc:
        data_set_test = data_set_test.shuffle(config.shuffle_pool_size). \
            padded_batch(config.batch_size, padded_shapes=([config.max_length], [config.max_length], [config.max_length], []))
    else:
        data_set_test = data_set_test.shuffle(config.shuffle_pool_size).repeat(). \
            padded_batch(config.batch_size, padded_shapes=([config.max_length], [config.max_length], [config.max_length], []))
    data_set_test_iter = data_set_test.make_one_shot_iterator()
    test_handle = sess.run(data_set_test_iter.string_handle())

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types,
                                                   data_set_train.output_shapes)
    input_ids, input_mask, segment_ids, labels = iterator.get_next()

    model = BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    saver = tf.train.Saver()
    output_layer = model.get_pooled_output()

    output = tf.layers.dense(output_layer, 3, name="output_dense")

    preds = tf.to_int32(tf.argmax(output, axis=-1))
    acc = tf.reduce_mean(tf.to_float(tf.equal(preds, labels)))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels))
    var_list = [v for v in tf.trainable_variables() if "output_dense" in v.name]

    num_train_steps = 4000
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.constant(value=0.001, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(
                            learning_rate,
                            global_step,
                            num_train_steps,
                            end_learning_rate=0.0,
                            power=1.0,
                            cycle=False)
    optimizer = AdamWeightDecayOptimizer(
                    learning_rate=learning_rate,
                    weight_decay_rate=0.01,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-6,
                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    train_op = optimizer.minimize(loss, global_step=global_step)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, pathname)

    print("starting training")
    for i in range(4000):
        train_feed = {handle: train_handle}
        sess.run(train_op, train_feed)
        if (i + 1) % 50 == 0:
            ttpred, ttacc, ttloss = sess.run([preds, acc, loss], train_feed)
            print("Generation train {} : acc: {}  loss: {} ".format(i, ttacc, ttloss))
        if (i + 1) % 100 == 0:
            tpred, tacc, tloss = sess.run([preds, acc, loss], {handle: test_handle})
            print("Generation test {} : acc: {}  loss: {} ".format(i, tacc, tloss))
        if (i + 1) % 2000 == 0:
            print("Generation train {} model saved ".format(i))
            saver.save(sess, os.path.join(model_save_path, model_name.format(model_choose)))
    saver.save(sess, os.path.join(model_save_path, model_name.format(model_choose)))
    print(1)


