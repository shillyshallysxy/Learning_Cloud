import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers


class LSTM_Model():
    def __init__(self, input_dim, lstm_dim_=100, hidden_dim_=50, num_tags_=1, lr_=0.001, batch_size_=128):
        self.lstm_dim = lstm_dim_
        self.hidden_dim = hidden_dim_
        self.lr = lr_
        self.num_tags = num_tags_
        self.initializer = initializers.xavier_initializer()
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        self.max_steps = tf.placeholder(dtype=tf.int32, shape=[None, ], name='seq_length')
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, input_dim], name='x_input')

        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, None], name='y_target')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, ],  name='sequence_length')
        self.batch_size = batch_size_
        self.num_steps = tf.shape(self.x_input)[-2]
        self.logits = self.project_layer(self.lstm_layer())

        self.loss = self.loss_layer(self.logits)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def lstm_layer(self):
        with tf.variable_scope("char_bilstm"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
            (outputs, outputs_state) = tf.nn.dynamic_rnn(
                lstm_fw_cell, self.x_input, sequence_length=self.sequence_length,
                initial_state=lstm_fw_cell.zero_state(self.batch_size, dtype=tf.float32),
                dtype=tf.float32)
        x_in_ = outputs
        # batch_size*n_steps*lstm_dim
        return x_in_

    def project_layer(self, x_in_):
        with tf.variable_scope("project"):
            with tf.variable_scope("hidden"):
                w_tanh = tf.get_variable("w_tanh", [self.lstm_dim, self.hidden_dim], initializer=self.initializer,
                                         regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_tanh = tf.get_variable("b_tanh", [self.hidden_dim], initializer=tf.zeros_initializer())
                x_in_ = tf.reshape(x_in_, [-1, self.lstm_dim])
                # output = tf.nn.dropout(tf.tanh(tf.add(tf.matmul(x_in_, w_tanh), b_tanh)), self.dropout)
                output = tf.nn.dropout(tf.tanh(tf.add(tf.matmul(x_in_, w_tanh), b_tanh)), self.dropout)
            with tf.variable_scope("output"):
                w_out = tf.get_variable("w_out", [self.hidden_dim, self.num_tags], initializer=self.initializer,
                                        regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_out = tf.get_variable("b_out", [self.num_tags], initializer=tf.zeros_initializer())
                pred_ = tf.add(tf.matmul(output, w_out), b_out)
                logits_ = tf.reshape(pred_, [-1, self.num_steps], name='logits')
        return logits_

    def project_layer_single(self, x_in_):
        with tf.variable_scope("project"):
            with tf.variable_scope("output"):
                w_out = tf.get_variable("w_out", [self.lstm_dim, self.num_tags], initializer=self.initializer,
                                        regularizer=tf.contrib.layers.l2_regularizer(0.001))
                b_out = tf.get_variable("b_out", [self.num_tags], initializer=tf.zeros_initializer())
                x_in_ = tf.reshape(x_in_, [-1, self.lstm_dim])
                pred_ = tf.add(tf.matmul(x_in_, w_out), b_out)
                logits_ = tf.reshape(pred_, [-1, self.num_steps], name='logits')
        return logits_

    def ms_error(self, labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def loss_layer(self, project_logits):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(project_logits, [-1], name='reshape_pred')],
            [tf.reshape(self.y_target, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.num_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            cost = tf.div(tf.reduce_sum(losses, name='losses_sum'),
                          self.batch_size,
                          name='average_cost')
        return cost
