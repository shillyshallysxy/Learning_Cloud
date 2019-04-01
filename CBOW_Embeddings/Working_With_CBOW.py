#  the Word2Vec Algorithm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from CBOW_Embeddings import text_helpers
import time

from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()

data_folder_name = '..\\temp'
data_path_name = 'cn_nlp\\translate'
vocab_name_cn = 'translate_cn.pkl'
vocab_name_en = 'translate_en.pkl'
train_record_name = 'translate_cn_en_train.tfrecord'
test_record_name = 'translate_cn_en_test.tfrecord'

vocab_path_cn = os.path.join(data_folder_name, data_path_name, vocab_name_cn)
vocab_path_en = os.path.join(data_folder_name, data_path_name, vocab_name_en)
model_save_path = os.path.join(data_folder_name, data_path_name, 'translate.ckpt')
model_log_path = os.path.join(data_folder_name, data_path_name, 'model_log.txt')

with open(vocab_path_cn, 'rb') as f:
    word_dict_cn = pickle.load(f)

with open(vocab_path_en, 'rb') as f:
    word_dict_en = pickle.load(f)

# Start a graph session
sess = tf.Session()

word_dict = word_dict_cn
# Declare model parameters
batch_size = 256
shuffle_pool_size = 2000
embedding_size = 100
vocabulary_size = len(word_dict)
generations = 20000
model_learning_rate = 0.001

num_sampled = int(batch_size/2)    # Number of negative examples to sample.
window_size = 3       # How many words to consider left and right.
name = 'skipgram'
# Add checkpoints to training
save_embeddings_every = 2000
print_valid_every = 2000
print_loss_every = 100


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text": tf.VarLenFeature(tf.int64),
                                                              "label": tf.VarLenFeature(tf.int64),
                                                              "text_length": tf.FixedLenFeature([], tf.int64),
                                                              "label_length": tf.FixedLenFeature([], tf.int64)})
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    text_ = tf.sparse_tensor_to_dense(features["text"])
    label_ = tf.sparse_tensor_to_dense(features["label"])
    text_lens_ = tf.cast(features["text_length"], tf.int32)
    label_lens_ = tf.cast(features["label_length"], tf.int32)
    return text_, label_, text_lens_, label_lens_


def get_dataset(record_name_):
    record_path_ = os.path.join(data_folder_name, data_path_name, record_name_)
    data_set_ = tf.data.TFRecordDataset(record_path_)
    return data_set_.map(__parse_function)


data_set_train = get_dataset(train_record_name)
data_set_train = data_set_train.shuffle(shuffle_pool_size).repeat().\
    padded_batch(batch_size, padded_shapes=([None], [None], [], []))
data_set_train_iter = data_set_train.make_one_shot_iterator()
train_handle = sess.run(data_set_train_iter.string_handle())

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types, data_set_train.output_shapes)
# _, texts, _, texts_length = iterator.get_next()
texts, _, texts_length, _ = iterator.get_next()

print('Creating Model')
# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings_en')

# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                              stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Create data/target placeholders
# x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*window_size])
if name == 'cbow':
    x_inputs = tf.placeholder(tf.int32, shape=[batch_size, window_size*2])
else:
    x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])

# Lookup the word embedding
# Add together window embeddings:
if name == 'cbow':
    embed = tf.zeros([batch_size, embedding_size])
    for element in range(2*window_size):
        embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])
else:
    embed = tf.nn.embedding_lookup(embeddings, x_inputs)

# Get loss from prediction
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))
                                     
# Create optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=model_learning_rate).minimize(loss)

# Cosine similarity between words
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm

# valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman', 'one', 'good', 'death']
valid_words = ['爱', '恨', '乐', '悲', '男', '女', '一', '好', '死']
valid_examples = [word_dict[x] for x in valid_words]
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Create model saving operation
saver = tf.train.Saver()

#Add variable initializer.
init = tf.global_variables_initializer()
sess.run(init)
model_checkpoint_path = os.path.join(data_folder_name, data_path_name, name + '_embeddings_cn.ckpt')
if False:
    print('retraining')
    saver.restore(sess, model_checkpoint_path)
# Run the CBOW model.
print('Starting Training')
loss_vec = []
loss_x_vec = []
word_dict_rev = dict(zip(word_dict.values(), word_dict.keys()))
for i in range(generations):
    text_data, text_data_length = sess.run([texts, texts_length], {handle: train_handle})
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, text_data_length, batch_size,
                                                                  window_size, name)
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)

    # Return the loss
    if (i+1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))

    if (i+1) % print_valid_every == 0:
        sim = sess.run(similarity)
        for j in range(len(valid_words)):
            valid_word = word_dict_rev[valid_examples[j]]
            top_k = 10  # number of nearest neighbors
            nearest = (-sim[j, 0:int(vocabulary_size * 0.8)]).argsort()[1:top_k + 1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dict_rev[nearest[k]]
                log_str = '{} {},'.format(log_str, close_word)
            print(log_str)
    # Save dictionary + embeddings
    if (i + 1) % save_embeddings_every == 0:
        # Save embeddings

        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))

# Start logistic model-------------------------
max_words = 20
logistic_batch_size = 500