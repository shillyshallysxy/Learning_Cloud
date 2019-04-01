import tensorflow as tf


class EncoderDecoderModel(object):
    def __init__(self, word2num_dict, config, encoder_embeddings=None, decoder_embeddings=None):
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.useTeacherForcing = config.useTeacherForcing
        self.useBeamSearch = config.useBeamSearch
        self.learning_rate = config.learning_rate
        self.encoder_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_input')
        self.encoder_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='encoder_length')
        self.decoder_output = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder_output')
        self.decoder_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='decoder_length')
        # with tf.variable_scope('embeddings'):
        #     self.encoder_embeddings = tf.get_variable(initializer=encoder_embeddings, name='embeddings_cn')
        #     self.decoder_embeddings = tf.get_variable(initializer=decoder_embeddings, name='embeddings_en')
        if encoder_embeddings is None or decoder_embeddings is None:
            self.encoder_embeddings = tf.Variable(tf.truncated_normal([config.vocabulary_size_cn,
                                                                       config.embedding_size_cn], stddev=0.05),
                                                  name='embeddings_cn')
            self.decoder_embeddings = tf.Variable(tf.truncated_normal([config.vocabulary_size_en,
                                                                       config.embedding_size_en], stddev=0.05),
                                                  name='embeddings_en')
        else:
            self.encoder_embeddings = encoder_embeddings
            self.decoder_embeddings = decoder_embeddings
        self.batch_size = tf.shape(self.encoder_input)[0]
        self.word2num_dict = word2num_dict  # decoder
        outputs_temp, states_temp = self.build_encoder()
        self.out, self.loss, self.train_op = self.build_decoder(outputs_temp, states_temp)

    def build_encoder(self):
        with tf.variable_scope('encoder_layer'):
            fw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            if self.num_layers > 1:
                fw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(self.hidden_size)
                                                       for _ in range(self.num_layers)])
                bw_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(self.hidden_size)
                                                       for _ in range(self.num_layers)])
            embed_encoder_input = tf.nn.embedding_lookup(self.encoder_embeddings, self.encoder_input)
            (encoder_outputs, encoder_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell, cell_bw=bw_cell, inputs=embed_encoder_input,
                sequence_length=self.encoder_length, dtype=tf.float32, time_major=False)
            encoder_outputs_u = tf.concat(encoder_outputs, axis=-1)
            # encoder_outputs_u = tf.add(encoder_outputs[0], encoder_outputs[1])
            if self.num_layers > 1:
                encoder_state_u = tf.concat(encoder_state, axis=-1)[-1]
                # encoder_state_u = tf.add(encoder_state[0], encoder_state[1])[-1]
            else:
                encoder_state_u = tf.concat(encoder_state, axis=-1)
                # encoder_state_u = tf.add(encoder_state[0], encoder_state[1])
        return encoder_outputs_u, encoder_state_u

    def build_decoder(self, encoder_outputs, encoder_states):
        with tf.variable_scope('decoder_layer'):
            tokens_begin = tf.ones([self.batch_size], dtype=tf.int32, name='tokens_begin')*self.word2num_dict['_BEGIN']
            # self.decoder_length = tf.add(self.decoder_length, 1)
            if self.useTeacherForcing:
                decoder_inputs = tf.concat([tf.reshape(tokens_begin, [-1, 1]), self.decoder_output[:, :-1]], 1)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    tf.nn.embedding_lookup(self.decoder_embeddings, decoder_inputs), self.decoder_length)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.decoder_embeddings,
                                                                  tokens_begin, self.word2num_dict["_EOS"])
            decoder_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size*2)
            if self.useBeamSearch > 1:
                tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.useBeamSearch)
                tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.encoder_length,
                                                                      multiplier=self.useBeamSearch)
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.hidden_size,
                                                                           memory=tiled_encoder_outputs,
                                                                           memory_sequence_length=tiled_sequence_length)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_states,
                                                                          multiplier=self.useBeamSearch)
                tiled_decoder_initial_state = decoder_cell.zero_state(
                    batch_size=self.batch_size * self.useBeamSearch, dtype=tf.float32)
                tiled_decoder_initial_state = tiled_decoder_initial_state.clone(
                    cell_state=tiled_encoder_final_state)
                decoder_initial_state = tiled_decoder_initial_state
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, self.decoder_embeddings, tokens_begin,
                                                               self.word2num_dict["_EOS"], decoder_initial_state,
                                                               beam_width=self.useBeamSearch,
                                                               output_layer=tf.layers.Dense(len(self.word2num_dict)))
            else:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(  # 或者LuongAttention
                    num_units=self.hidden_size, memory=encoder_outputs, memory_sequence_length=self.encoder_length)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
                decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_states)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                          output_layer=tf.layers.Dense(len(self.word2num_dict)))

            decoder_outputs, decoder_state, final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(self.decoder_length))

            if self.useBeamSearch > 1:
                out = decoder_outputs.predicted_ids[:, :, 0]
                loss = None
                train_op = None
            else:
                decoder_logits = decoder_outputs.rnn_output
                out = tf.argmax(decoder_logits, 2)

                sequence_mask = tf.sequence_mask(self.decoder_length, dtype=tf.float32)
                loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits, targets=self.decoder_output,
                                                        weights=sequence_mask)
                train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            # pred_out = tf.Variable(out, name='pred_out')
            return out, loss, train_op
