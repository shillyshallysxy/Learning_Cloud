import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers


class LSTM_Model():
    def __init__(self, input_dim, embeddings_length, embedding_size, lstm_dim_=100, hidden_dim_=50, num_tags_=1, lr_=0.001):
        self.lstm_dim = lstm_dim_
        self.hidden_dim = hidden_dim_
        self.lr = lr_
        self.num_tags = num_tags_
        self.initializer = initializers.xavier_initializer()
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, input_dim], name='x_input')
        self.num_steps = tf.shape(self.x_input)[-2]
        self.batch_size = tf.shape(self.x_input)[0]
        self.city_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='city_input')
        self.embeddings = tf.Variable(tf.random_uniform([embeddings_length, embedding_size], -1, 1, dtype=tf.float32,
                                                        ), name='embeddings')
        embeded = tf.nn.embedding_lookup(self.embeddings, self.city_input)  # batch_size*num_steps*embedding_size
        city_var_dim = 20
        self.city_var = tf.Variable(tf.truncated_normal([embedding_size, city_var_dim], stddev=0.05, dtype=tf.float32), name="city_w")
        self.city_bias = tf.Variable(tf.truncated_normal([city_var_dim], stddev=0.05, dtype=tf.float32), name="city_b")
        city_output = tf.reshape(tf.nn.relu(tf.add(tf.matmul(tf.reshape(embeded, [-1, embedding_size]), self.city_var), self.city_bias)),
                                 [-1, self.num_steps, city_var_dim])
        # city_output = embeded
        self.concated_input = tf.concat([self.x_input, city_output], axis=2)  # batch_size*num_steps*(input_dim+10)
        self.y_target = tf.placeholder(dtype=tf.float32, shape=[None, None], name='y_target')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='sequence_length')
        self.logits = self.project_layer(self.lstm_layer())

        self.loss = self.loss_layer()
        trainable_var = tf.trainable_variables()
        var_list = [x for x in tf.trainable_variables() if "embeddings" in x.name or "city" in x.name]
        # print(var_list)
        # self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=var_list)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def lstm_layer(self):
        with tf.variable_scope("char_bilstm"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
            (outputs, outputs_state) = tf.nn.dynamic_rnn(
                lstm_fw_cell, self.concated_input, sequence_length=self.sequence_length,
                # initial_state=lstm_fw_cell.zero_state(self.batch_size, dtype=tf.float32),
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

    def loss_layer(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.logits, [-1], name='reshape_pred')],
            [tf.reshape(self.y_target, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.num_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            cost = tf.reduce_sum(losses, name='losses_sum')
        return cost
