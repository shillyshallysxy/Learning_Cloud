from Transformer.transformer_utils import *
import tensorflow as tf


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
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
