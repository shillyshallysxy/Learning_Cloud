import tensorflow as tf
import math
import numpy as np


def gelu(inputs):
    """
    gelu: https://arxiv.org/abs/1606.08415
    :param inputs: [Tensor]
    :return: [Tensor] outputs after activation
    """
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2 / np.pi) * (inputs + 0.044715 * tf.pow(inputs, 3))))
    return inputs * cdf


def get_activation(activation_name):
    """
    get activate function
    :param activation_name: [Tensor]
    :return: [Function] activation function
    """
    if activation_name is None:
        return gelu
    else:
        act = activation_name.lower()
        if act == "relu":
            return tf.nn.relu
        elif act == "gelu":
            return gelu
        elif act == "tanh":
            return tf.tanh
        else:
            raise ValueError("Unsupported activation: %s" % act)


def reshape_to_matrix(inputs):
    """
    Reshapes inputs tensor to a rank 2 tensor.
    :param inputs: [Tensor]
    :return: [Tensor] tensor with two dimensions
    """
    ndims = inputs.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = {}".format(inputs.shape))
    if ndims == 2:
        return inputs

    width = inputs.shape[-1]
    output_tensor = tf.reshape(inputs, [-1, width])
    return output_tensor


def group_norm(inputs: tf.Tensor, epsilon=1e-8, scope="layer_normalization", reuse=None):
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
    N = tf.shape(inputs)[0]
    T = tf.shape(inputs)[1]
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
    position = tf.to_float(tf.range(start_index, length))
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(min_timescale) / float(max_timescale)) /
                               (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.expand_dims(signal, axis=0)
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
        lookup_table = tf.concat((tf.zeros(shape=[1, channels], dtype=tf.float32),
                                  lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * math.sqrt(channels)

    return outputs


def multi_head_attention(from_tensor: tf.Tensor,  to_tensor: tf.Tensor, channels=None, num_units=None, num_heads=8,
                         dropout_rate=0, is_training=True, causality=False, scope="multihead_attention",
                         activation=None, reuse=None):
    """
    multihead attention
    :param from_tensor: [Tensor] query matrix
    :param to_tensor: [Tensor] key matrix
    :param channels: [Int] channel of last dimension of output
    :param num_units: [Int] channel size of matrix Q, K, V
    :param num_heads: [Int] head number of attention
    :param dropout_rate: [Float] dropout rate when 0 means no dropout
    :param is_training: [Boolean] whether it is training, If true, use dropout
    :param causality: [Boolean] If true, units that reference the future are masked
    :param scope: [String] name of "variable_scope"
    :param activation: [String] name of activate function
    :param reuse: [Boolean] tf parameter reuse
    :return: [Tensor] outputs after multihead self attention with shape of "batch_size * max_length * (channels*num_heads)"
    """
    with tf.variable_scope(scope, reuse=reuse):
        if channels is None:
            channels = from_tensor.get_shape().as_list()[-1]
        if num_units is None:
            num_units = channels//num_heads
        activation_fn = get_activation(activation)
        # shape [batch_size, max_length, channels*num_heads]
        query_layer = tf.layers.dense(from_tensor, num_units * num_heads, activation=activation_fn)
        key_layer = tf.layers.dense(to_tensor, num_units * num_heads, activation=activation_fn)
        value_layer = tf.layers.dense(to_tensor, num_units * num_heads, activation=activation_fn)

        # shape [batch_size*num_heads, max_length, channels]
        query_layer_ = tf.concat(tf.split(query_layer, num_heads, axis=2), axis=0)
        key_layer_ = tf.concat(tf.split(key_layer, num_heads, axis=2), axis=0)
        value_layer_ = tf.concat(tf.split(value_layer, num_heads, axis=2), axis=0)

        # shape = [batch_size*num_heads, max_length, max_length]
        attention_scores = tf.matmul(query_layer_, tf.transpose(key_layer_, [0, 2, 1]))
        # Scale
        attention_scores = tf.multiply(attention_scores, 1.0 / tf.sqrt(float(channels)))
        # attention masks
        attention_masks = tf.sign(tf.reduce_sum(to_tensor, axis=-1))
        attention_masks = tf.tile(attention_masks, [num_heads, 1])
        attention_masks = tf.tile(tf.expand_dims(attention_masks, axis=1), [1, tf.shape(from_tensor)[1], 1])
        neg_inf_matrix = tf.multiply(tf.ones_like(attention_scores), (-math.pow(2, 32) + 1))
        attention_scores = tf.where(tf.equal(attention_masks, 0), neg_inf_matrix, attention_scores)

        if causality:
            diag_vals = tf.ones_like(attention_scores[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()

            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(attention_scores)[0], 1, 1])
            neg_inf_matrix = tf.multiply(tf.ones_like(masks), (-math.pow(2, 32) + 1))
            attention_scores = tf.where(tf.equal(masks, 0), neg_inf_matrix, attention_scores)

        # attention probability
        attention_probs = tf.nn.softmax(attention_scores)

        # query mask
        query_masks = tf.sign(tf.abs(tf.reduce_sum(from_tensor, axis=-1)))
        # shape = [N*h, T_q]
        query_masks = tf.tile(query_masks, [num_heads, 1])
        # shape = [N*h, T_q, T_k]
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(to_tensor)[1]])

        attention_probs *= query_masks

        # dropout
        attention_probs = tf.layers.dropout(attention_probs, rate=dropout_rate,
                                            training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(attention_probs, value_layer_)
        # shape [batch_size, max_length, channels*num_heads]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # reshape to from tensor
        outputs = tf.layers.dense(outputs, channels, activation=activation_fn)
        # Residual connection
        outputs += from_tensor
        # group normalization
        outputs = group_norm(outputs)
    return outputs


def feed_forward(inputs, channels, hidden_dims=None, scope="multihead_attention", activation=None, reuse=None):
    """
    :param inputs: [Tensor] with first dimension of "batch_size"
    :param channels: [Int] Embedding size
    :param hidden_dims: [List] hidden dimensions
    :param scope: [String] name of "variable_scope"
    :param activation: [String] name of activate function
    :param reuse: [Boolean] tf parameter reuse
    :return: [Tensor] outputs after feed forward with shape of "batch_size * max_length * channels"
    """
    if hidden_dims is None:
        hidden_dims = []
    with tf.variable_scope(scope, reuse=reuse):
        activation_fn = get_activation(activation)
        outputs = inputs
        for hidden_dim in hidden_dims:
            params = {"inputs": outputs, "num_outputs": hidden_dim, "activation_fn": activation_fn}
            outputs = tf.contrib.layers.fully_connected(**params)
        params = {"inputs": outputs, "num_outputs": channels, "activation_fn": activation_fn}
        outputs = tf.contrib.layers.fully_connected(**params)
        outputs += inputs
        outputs = group_norm(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    """
    :param inputs: [Tensor]
    :param epsilon: [Float] Smoothing rate
    :return: [Tensor] outputs after smoothing
    """
    return ((1 - epsilon) * inputs) + (epsilon / inputs.get_shape().as_list()[-1])
