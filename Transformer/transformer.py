import tensorflow as tf
import math


def group_norm(inputs: tf.Tensor, epsilon: float, scope: str, reuse: bool)->tf.Tensor:
    """
    layer normalization
    :param inputs: [Tensor] with first dimension of "batch_size"
    :param epsilon: [Float] a number for preventing ZeroDivision
    :param scope: [String] name of "variable_scope"
    :param reuse: [Boolean] tf parameter reuse
    :return: [Tensor] outputs after normalized
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)
        outputs = gamma * normalized + beta
    return outputs


def get_positional_encoding(inputs, channels, scope="positional_embedding", reuse=None):
    """
    positional encoding
    :param inputs: [Tensor] with dimension of "batch_size * max_length"
    :param channels: [Int] Embedding size
    :param scope: [String] name of "variable_scope"
    :param reuse: [Boolean] tf parameter reuse
    :return: [Tensor] outputs after positional encoding
    """
    N, T = inputs.get_shape().as_list()

    with tf.variable_scope(scope, reuse=reuse):
        position_signal = get_timing_signal_1d(T, channels)
        position_signal = tf.tile(position_signal, [N, 1, 1])
    return position_signal


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """
    positional encoding的方法
    :param length: [Int] max_length size
    :param channels: [Int] Embedding size
    :param min_timescale: [Float]
    :param max_timescale: [Float]
    :param start_index: [Int] index of first position
    :return: [Tensor] positional encoding of shape "1 * length * channels"
    """
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(min_timescale) / float(max_timescale)) /
                               (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def get_embedding(inputs, vocab_size, channels, scale=True, scope="embedding", reuse=None):
    """
    embedding
    :param inputs: [Tensor] with first dimension of "batch_size"
    :param vocab_size: [Int] Vocabulary size
    :param channels: [Int] Embedding size
    :param scale: [Boolean] If True, the output will be multiplied by sqrt num_units
    :param scope: [String] name of "variable_scope"
    :param reuse: [Boolean] tf parameter reuse
    :return: [Tensor] outputs of embedding of sentence with shape of "batch_size * length * channels"
    """
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, channels],
                                       initializer=tf.contrib.layers.xavier_initializer())

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * math.sqrt(channels)

    return outputs


def multi_head_attention(queries,  keys,  channels,  num_heads,  dropout_rate=0,  is_training=True,
                         causality = False, scope = "multihead_attention", reuse = None):
    """
    multihead attention
    :param queries: [Tensor] query matrix
    :param keys: [Tensor] key matrix
    :param channels: [Int] Embedding size
    :param num_heads: [Int] head number of attention
    :param dropout_rate: [Float] dropout rate when 0 means no dropout
    :param is_training: [Boolean] whether it is training, If true, use dropout
    :param causality: [Boolean] If true, units that reference the future are masked
    :param scope: [String] name of "variable_scope"
    :param reuse: [Boolean] tf parameter reuse
    :return: [Tensor] outputs after multihead self attention with shape of "batch_size * max_length * channels"
    """
