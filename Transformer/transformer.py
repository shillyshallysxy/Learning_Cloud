from Transformer.transformer_utils import *
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood


class Transformer():
    def __init__(self, inputs, outputs, pm):

        self.inputs = tf.to_int32(inputs)
        self.outputs = tf.to_int32(outputs)
        # 在开头加上2（即2<STR>，将末尾的3<EOS>去掉
        self.decoder_input = tf.concat((tf.ones_like(self.outputs[:, :1]), self.outputs[:, :-1]), -1)
        self.vocab_size_en = pm.vocab_size_en
        self.vocab_size_de = pm.vocab_size_de
        self.channels = pm.channels
        self.num_heads = pm.num_heads
        self.dropout_rate = pm.dropout_rate
        self.is_training = pm.is_training
        self.num_layer = pm.layer_num
        self.learning_rate = pm.learning_rate
        with tf.variable_scope("encoder"):
            self.encode = get_embedding(self.inputs, self.vocab_size_en, self.channels, scope="en_embed")
            self.encode += get_positional_encoding(self.inputs, self.channels, scope="en_pe")
            for i in range(self.num_layer):
                with tf.variable_scope("encoder_layer_{}".format(i)):
                    self.encode = multi_head_attention(self.encode, self.encode, self.channels,
                                                       num_heads=self.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=False)
                    self.encode = feed_forward(self.encode, self.channels)

        with tf.variable_scope("decoder"):
            self.decode = get_embedding(self.decoder_input, self.vocab_size_de, self.channels, scope="de_embed")
            self.decode += get_positional_encoding(self.decode, self.channels)
            for i in range(self.num_layer):
                with tf.variable_scope("decoder_layer_{}".format(i)):
                    self.decode = multi_head_attention(self.decode, self.decode, self.channels,
                                                       num_heads=self.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=True,
                                                       scope="self_attention")
                    self.decode = multi_head_attention(self.decode, self.encode, self.channels,
                                                       num_heads=self.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=False,
                                                       scope="encoder_decoder_attention")
                    self.decode = feed_forward(self.decode, self.channels)
        self.logits = tf.layers.dense(self.decode, self.vocab_size_de)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        self.istarget = tf.to_float(tf.not_equal(self.outputs, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.outputs)) * self.istarget) / (
            tf.reduce_sum(self.istarget))
        if self.is_training:
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.outputs)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=0.9, beta2=0.98, epsilon=1e-8)
            # self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            self.grads = self.optimizer.compute_gradients(self.mean_loss)
            if pm.is_transfer_learning:
                var_list = tf.trainable_variables()
                layer_name_list = ["encoder_layer_" + str(i) for i in range(4)]
                var_list_ = [v for v in var_list if v.name.split("/")[1] in layer_name_list]
                for index, grad in enumerate(self.grads):
                    if grad[1] in var_list_:
                        self.grads[index] = (grad[0]*0.1, grad[1])
            self.train_op = self.optimizer.apply_gradients(self.grads, global_step=self.global_step)


class TransformerSNLIPool():

    def __init__(self, inputs1, inputs2, label, pm):
        self.inputs1 = tf.to_int32(inputs1)
        self.inputs2 = tf.to_int32(inputs2)
        self.target = tf.to_int32(label)
        self.vocab_size_en = pm.vocab_size_en
        self.channels = pm.channels
        self.num_heads = pm.num_heads
        self.dropout_rate = pm.dropout_rate
        self.is_training = pm.is_training
        self.num_layer = pm.layer_num
        self.learning_rate = pm.learning_rate
        with tf.variable_scope("encoder"):
            self.encode1 = get_embedding(self.inputs1, self.vocab_size_en, self.channels, scope="en_embed")
            self.encode1 += get_positional_encoding(self.inputs1, self.channels, scope="en_pe")
            for i in range(self.num_layer):
                with tf.variable_scope("encoder_layer_{}".format(i)):
                    self.encode1 = multi_head_attention(self.encode1, self.encode1, self.channels,
                                                        num_heads=self.num_heads,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        causality=False)
                    self.encode1 = feed_forward(self.encode1, self.channels)
                    self.max_pool1 = tf.layers.max_pooling1d(self.encode1, pool_size=pm.max_length, strides=1)
                    self.max_pool1 = tf.reshape(self.max_pool1, [-1, self.channels])
        with tf.variable_scope("encoder", reuse=True):
            self.encode2 = get_embedding(self.inputs2, self.vocab_size_en, self.channels, scope="en_embed", reuse=True)
            self.encode2 += get_positional_encoding(self.inputs2, self.channels, scope="en_pe", reuse=True)
            for i in range(self.num_layer):
                with tf.variable_scope("encoder_layer_{}".format(i)):
                    self.encode2 = multi_head_attention(self.encode2, self.encode2, self.channels,
                                                        num_heads=self.num_heads,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        causality=False, reuse=True)
                    self.encode2 = feed_forward(self.encode2, self.channels, reuse=True)
                    self.max_pool2 = tf.layers.max_pooling1d(self.encode2, pool_size=pm.max_length, strides=1)
                    self.max_pool2 = tf.reshape(self.max_pool2, [-1, self.channels])

        sub_pool = tf.subtract(self.max_pool1, self.max_pool2)
        mul_pool = tf.multiply(self.max_pool1, self.max_pool2)
        self.output = tf.concat([self.max_pool1, self.max_pool2, sub_pool, mul_pool], axis=1)
        self.output = tf.layers.dense(self.output, self.channels)

        self.output = tf.layers.dense(self.output, pm.num_tags)
        self.preds = tf.to_int32(tf.argmax(self.output, axis=-1))
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.preds, self.target)))
        if self.is_training:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.target))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


class TransformerSNLIClsB():

    def __init__(self, inputs1, inputs2, label, pm):
        self.inputs1 = tf.to_int32(inputs1)
        self.inputs2 = tf.to_int32(inputs2)
        self.target = tf.to_int32(label)
        self.vocab_size_en = pm.vocab_size_en
        self.channels = pm.channels
        self.num_heads = pm.num_heads
        self.dropout_rate = pm.dropout_rate
        self.is_training = pm.is_training
        self.num_layer = pm.layer_num
        self.learning_rate = pm.learning_rate
        # {'_PAD': 0, '_BEGIN': 1, '_EOS': 2, '_CLS': 3, '_SEP': 4, '_MASK': 5}
        self.inputs1 = tf.concat((tf.ones_like(self.inputs1[:, :1]) * 3, self.inputs1), axis=-1)
        self.inputs2 = tf.concat((tf.ones_like(self.inputs2[:, :1]) * 3, self.inputs2), axis=-1)
        with tf.variable_scope("encoder"):
            self.encode1 = get_embedding(self.inputs1, self.vocab_size_en, self.channels, scope="en_embed")
            self.encode1 += get_positional_encoding(self.inputs1, self.channels, scope="en_pe")
            for i in range(self.num_layer):
                with tf.variable_scope("encoder_layer_{}".format(i)):
                    self.encode1 = multi_head_attention(self.encode1, self.encode1, self.channels,
                                                        num_heads=self.num_heads,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        causality=False)
                    self.encode1 = feed_forward(self.encode1, self.channels)
                    self.encode_cls1 = tf.reshape(self.encode1[:, :1, :], [-1, self.channels])
        with tf.variable_scope("encoder", reuse=True):
            self.encode2 = get_embedding(self.inputs2, self.vocab_size_en, self.channels, scope="en_embed", reuse=True)
            self.encode2 += get_positional_encoding(self.inputs2, self.channels, scope="en_pe", reuse=True)
            for i in range(self.num_layer):
                with tf.variable_scope("encoder_layer_{}".format(i)):
                    self.encode2 = multi_head_attention(self.encode2, self.encode2, self.channels,
                                                        num_heads=self.num_heads,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        causality=False, reuse=True)
                    self.encode2 = feed_forward(self.encode2, self.channels, reuse=True)
                    self.encode_cls2 = tf.reshape(self.encode2[:, :1, :], [-1, self.channels])

        self.output = tf.concat([self.encode_cls1, self.encode_cls2], axis=1)
        self.output = tf.layers.dense(self.output, pm.num_tags)
        self.preds = tf.to_int32(tf.argmax(self.output, axis=-1))
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.preds, self.target)))
        if self.is_training:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.target))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


class TransformerSNLICls():
    def __init__(self, inputs1, inputs2, label, pm):
        self.inputs1 = tf.to_int32(inputs1)  # batch_size*max_length
        self.inputs2 = tf.to_int32(inputs2)
        self.target = tf.to_int32(label)
        self.vocab_size_en = pm.vocab_size_en
        self.channels = pm.channels
        self.num_heads = pm.num_heads
        self.dropout_rate = pm.dropout_rate
        self.is_training = pm.is_training
        self.num_layer = pm.layer_num
        self.learning_rate = pm.learning_rate
        # {'_PAD': 0, '_BEGIN': 1, '_EOS': 2, '_CLS': 3, '_SEP': 4, '_MASK': 5}
        self.inputs1 = tf.concat((tf.ones_like(self.inputs1[:, :1])*3, self.inputs1), axis=-1)
        self.inputs1 = tf.concat((self.inputs1, tf.ones_like(self.inputs1[:, :1])*4), axis=-1)
        self.inputs2 = tf.concat((self.inputs2, tf.ones_like(self.inputs2[:, :1])*4), axis=-1)

        with tf.variable_scope("encoder"):
            self.encode1 = get_embedding(self.inputs1, self.vocab_size_en, self.channels, scope="en_embed")
            self.test = get_embedding(self.inputs1, self.vocab_size_en, self.channels, scope="en_embed", reuse=True)
            self.encode2 = get_embedding(self.inputs2, self.vocab_size_en, self.channels, scope="en_embed", reuse=True)
            self.encode1 += get_positional_encoding(self.inputs1, self.channels, scope="en_pe")
            self.encode2 += get_positional_encoding(self.inputs2, self.channels, scope="en_pe", reuse=True)
            self.encode1 += get_seg_embedding(self.inputs1, self.channels, 1, scope="en_se")
            self.encode2 += get_seg_embedding(self.inputs2, self.channels, 2, scope="en_se", reuse=True)
            self.encode = tf.concat((self.encode1, self.encode2), axis=1)
            self.encode = tf.layers.dropout(self.encode, rate=self.dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))
            for i in range(self.num_layer):
                with tf.variable_scope("encoder_layer_{}".format(i)):
                    self.encode = multi_head_attention(self.encode, self.encode, self.channels,
                                                       num_heads=self.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=False)
                    self.encode = feed_forward(self.encode, self.channels)
            self.encode_cls = tf.reshape(self.encode[:, :1, :], [-1, self.channels])

        self.output = tf.layers.dense(self.encode_cls, pm.num_tags)
        self.preds = tf.to_int32(tf.argmax(self.output, axis=-1))
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.preds, self.target)))
        if self.is_training:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.target))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.grads = self.optimizer.compute_gradients(self.loss)
            if pm.is_transfer_learning:
                var_list = tf.trainable_variables()
                layer_name_list = ["encoder_layer_" + str(i) for i in range(4)]
                var_list_ = [v for v in var_list if v.name.split("/")[1] in layer_name_list]
                var_list_ += [v for v in var_list if "lookup_table" in v.name]
                for index, grad in enumerate(self.grads):
                    if grad[1] in var_list_:
                        self.grads[index] = (grad[0]*0.2, grad[1])
            self.train_op = self.optimizer.apply_gradients(self.grads, global_step=self.global_step)


class TransformerPOS():
    def __init__(self, inputs, outputs, seq_length, pm):
        self.inputs = tf.to_int32(inputs)
        self.outputs = tf.to_int32(outputs)
        self.seq_length = tf.to_int32(seq_length)
        self.vocab_size_en = pm.vocab_size_en
        self.channels = pm.channels
        self.num_heads = pm.num_heads
        self.dropout_rate = pm.dropout_rate
        self.is_training = pm.is_training
        self.num_layer = pm.layer_num
        self.learning_rate = pm.learning_rate
        self.trans = tf.get_variable("transitions", shape=[pm.num_tags, pm.num_tags],
                                     initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope("encoder"):
            self.encode = get_embedding(self.inputs, self.vocab_size_en, self.channels, scope="en_embed")
            self.encode += get_positional_encoding(self.inputs, self.channels, scope="en_pe")
            for i in range(self.num_layer):
                with tf.variable_scope("encoder_layer_{}".format(i)):
                    self.encode = multi_head_attention(self.encode, self.encode, self.channels,
                                                       num_heads=self.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=False)
                    self.encode = feed_forward(self.encode, self.channels)
        self.logits = tf.layers.dense(self.encode, pm.num_tags)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        if self.is_training:
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.preds, self.outputs)))
            with tf.variable_scope("crf_loss"):
                log_likelihood, trans = crf_log_likelihood(inputs=self.logits, tag_indices=self.outputs,
                                                           transition_params=self.trans,
                                                           sequence_lengths=self.seq_length)
                self.loss = tf.reduce_mean(-log_likelihood)
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)