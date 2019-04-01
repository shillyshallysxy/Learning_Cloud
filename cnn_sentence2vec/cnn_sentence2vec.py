import pickle as pkl
import numpy as np
import text_helpers
from nltk.corpus import stopwords
import os
import tensorflow as tf
freq_table_path = 'movie_vocab_freq.pkl'
word_dict_path = 'movie_vocab.pkl'
ckp_name = 'skipgram_movie_embeddings.ckpt'
data_folder_name = 'G:\\python\DeepLearning\learn_tensorflow\part7\\temp'
embedding_size = 200
batch_size = 1
n_gram = 2
num_channels = 1
conv_features = 256
sentence_size = 128
sess = tf.Session()
# 词典词频嵌入载入
with open(os.path.join(data_folder_name, word_dict_path), 'rb') as f:
    word_dict = pkl.load(f)
print('完成词字典载入')
embeddings = tf.Variable(tf.random_uniform([len(word_dict), embedding_size], -1, 1))
saver = tf.train.Saver({'embeddings': embeddings})
ckp_path = os.path.join(data_folder_name, ckp_name)
saver.restore(sess, ckp_path)
embed = sess.run(embeddings)
print('完成词嵌入载入')
# 载入句子
stops = stopwords.words('english')
texts, target = text_helpers.load_movie_data(data_folder_name)
print('完成文本载入')
texts = text_helpers.normalize_text(texts, stops)
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]
texts = text_helpers.text_to_numbers(texts, word_dict)
sentences_embed = []
for x in texts:
    sentence_embed = np.array([embed[y] for y in x])
    sentences_embed.append(sentence_embed)
sentences_embed = np.array(sentences_embed)
print('完成文本处理')
# Need to keep the indices sorted to keep track of document index
sentence_vec = sentences_embed
train_indices = np.sort(np.random.choice(len(target), round(0.8 * len(target)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
texts_train = np.array([x for ix, x in enumerate(sentence_vec) if ix in train_indices])
texts_test = np.array([x for ix, x in enumerate(sentence_vec) if ix in test_indices])
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
print('完成shuffle')
x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, embedding_size],
                         name='sentence_concatenate')
y_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target')

conv1_weight = tf.Variable(tf.truncated_normal(shape=[n_gram, embedding_size, num_channels, conv_features],
                                               stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros(shape=[conv_features], dtype=tf.float32))

full1_weight = tf.Variable(tf.truncated_normal(shape=[conv_features, sentence_size],
                                               stddev=0.1, dtype=tf.float32))
full1_bais = tf.Variable(tf.zeros(shape=[1, sentence_size], dtype=tf.float32))

W_weight = tf.Variable(tf.truncated_normal(shape=[sentence_size, 1],
                                           stddev=0.1, dtype=tf.float32))
b_bais = tf.Variable(tf.zeros(shape=[1, 1], dtype=tf.float32))

conv1 = tf.nn.conv2d(tf.expand_dims(x_input, 3), conv1_weight, strides=[1, 1, 1, 1], padding='VALID')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
max_pool = tf.reduce_max(relu1, axis=1, keepdims=True)
full1_input = tf.squeeze(max_pool, axis=[1, 2])
sentence_vec_output = tf.add(tf.matmul(full1_input, full1_weight), full1_bais)
model_output = tf.add(tf.matmul(sentence_vec_output, W_weight), b_bais)
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_step = optimizer.minimize(loss, var_list=[conv1_weight, conv1_bias, full1_weight, full1_bais, W_weight, b_bais])
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, tf.cast(y_target, tf.float32)), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)
init = tf.global_variables_initializer()
sess.run(init)
print('Starting Logistic Doc2Vec Model Training')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(len(train_indices), size=batch_size)
    rand_x = texts_train[rand_index][0]
    rand_x = np.expand_dims(rand_x, axis=0)
    # rand_x = np.hstack((rand_x, np.transpose([rand_x_doc_indices])))
    rand_y = np.transpose([target_train[rand_index]])
    feed_dict = {x_input: rand_x, y_target: rand_y}
    sess.run(train_step, feed_dict=feed_dict)
    if (i + 1) % 100 == 0:
        print('loss:', sess.run(loss, feed_dict=feed_dict))
    # if (i + 1) % 100 == 0:
    #     rand_index_test = np.random.choice(len(test_indices), size=batch_size)
    #     rand_x_test = texts_test[rand_index_test][0]
    #     rand_x_test = np.expand_dims(rand_x_test, axis=0)
    #     # Append review index at the end of text data
    #     rand_y_test = np.transpose([target_test[rand_index_test]])
    #
    #     test_feed_dict = {x_input: rand_x_test, y_target: rand_y_test}
    #
    #     i_data.append(i + 1)
    #
    #     train_loss_temp = sess.run(loss, feed_dict=feed_dict)
    #     train_loss.append(train_loss_temp)
    #
    #     test_loss_temp = sess.run(loss, feed_dict=test_feed_dict)
    #     test_loss.append(test_loss_temp)
    #
    #     train_acc_temp = sess.run(accuracy, feed_dict=feed_dict)
    #     train_acc.append(train_acc_temp)
    #
    #     test_acc_temp = sess.run(accuracy, feed_dict=test_feed_dict)
    #     test_acc.append(test_acc_temp)
    # if (i + 1) % 500 == 0:
    #     acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
    #     acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
    #     print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
    #         *acc_and_loss))

print('1')

