import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import numpy as np
from tensorflow.python.framework import ops

embedding_size = 200   # Word embedding size
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

data_folder_name = 'temp'
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

vocabulary_size = len(word_dict_en)
sess = tf.Session()
word_dict = word_dict_en
word_dict_rev = dict(zip(word_dict.values(), word_dict.keys()))
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1, 1))
model_ckpt_path = os.path.join(data_folder_name, data_path_name, 'cbow'+'_embeddings_en.ckpt')
saver = tf.train.Saver({'embeddings_en': embeddings})
saver.restore(sess, model_ckpt_path)
embed_m = sess.run(embeddings)
U, s, Vh = np.linalg.svd(embed_m, full_matrices=False)
x_max = max(U[:, 0])
x_min = min(U[:, 0])
y_max = max(U[:, 1])
y_min = min(U[:, 1])
# print('s_matrix: ', s)
for ix, word in enumerate(word_dict.keys()):

    if ix < 1000:
        plt.text(U[ix, 0], U[ix, 1], word)
plt.ylim([y_min, y_max])
plt.xlim([x_min, x_max])
plt.show()


valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman', 'one', 'good', 'death']
valid_examples = [word_dict[x] for x in valid_words]
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
sim = sess.run(similarity)
for j in range(len(valid_words)):
    valid_word = word_dict_rev[valid_examples[j]]
    top_k = 10  # number of nearest neighbors
    nearest = (-sim[j, 0:int(vocabulary_size*0.8)]).argsort()[1:top_k+1]
    log_str = "Nearest to {}:".format(valid_word)
    for k in range(top_k):
        close_word = word_dict_rev[nearest[k]]
        log_str = '{} {},' .format(log_str, close_word)
    print(log_str)
exit()
