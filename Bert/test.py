import tensorflow as tf
from Bert.modeling import *
import os
from Bert.tokenization import *
from Bert.optimization import *
pathname = "G:\\tmp\\toxic_model\\model.ckpt-26791"  # 模型地址
bert_config = BertConfig.from_json_file("uncased_L-12_H-768_A-12/bert_config.json")  # 配置文件地址。
vocab_path = "uncased_L-12_H-768_A-12/vocab.txt"


def __parse_function(serial_exmp):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([128], tf.int64),
        "input_mask": tf.FixedLenFeature([128], tf.int64),
        "segment_ids": tf.FixedLenFeature([128], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    features = tf.parse_single_example(serial_exmp, features=name_to_features)
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    texts_ = tf.to_int32(features["input_ids"])
    mask_ = tf.to_int32(features["input_mask"])
    seg_ = tf.to_int32(features["segment_ids"])
    label_ = tf.to_int32(features["label_ids"])
    return texts_, mask_, seg_, label_


def get_dataset(record_name_):
    record_path_ = "G:\\tmp\\toxic_model\\" + record_name_
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

    data_set_train = get_dataset("train.tf_record")
    data_set_train = data_set_train.shuffle(config.shuffle_pool_size).repeat(). \
        padded_batch(config.batch_size, padded_shapes=([config.max_length], [config.max_length], [config.max_length], []))
    data_set_train_iter = data_set_train.make_one_shot_iterator()
    train_handle = sess.run(data_set_train_iter.string_handle())

    data_set_test = get_dataset(os.path.join("test.tf_record"))
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
    input_ids, input_mask, segment_ids, labels_ids = iterator.get_next()

    # 创建bert模型
    model = BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False  # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
    )
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [4, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [4], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if True:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels_ids, depth=4, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels_ids)))

    # bert模型参数初始化的地方
    use_tpu = False
    # 获取模型中所有的训练参数。
    tvars = tf.trainable_variables()
    # 加载BERT模型

    (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, pathname)

    tf.train.init_from_checkpoint(pathname, assignment_map)
    # [1] train_examples
    num_train_steps = int(len([1]) / config.batch_size * config.num_train_epochs)
    num_warmup_steps = int(num_train_steps * config.warmup_proportion)
    init_lr = 5e-5
    train_op = create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, False)

    tf.logging.info("**** Trainable Variables ****")
    # 打印加载模型的参数
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        print(sess.run(accuracy, {handle: train_handle}))
    save_var_list = [v for v in tvars if "embeddings" in v.name ]
    save_var_list += [v for v in tvars if v.name.startswith("bert/encoder/layer")]
    print(1)


