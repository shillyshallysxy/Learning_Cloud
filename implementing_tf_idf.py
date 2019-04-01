import tensorflow as tf
import csv
import numpy as np
import os
import string
import requests
import io
import nltk
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()
batch_size = 200
max_features = 1000

# 读取短信数据集（如果已经下载过了，就读取本地数据不再下载）
save_file_name = 'temp_spam_data.csv'
if os.path.isfile(save_file_name):
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            text_data.append(row)
else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x) >= 1]
    with open(save_file_name, 'w', newline='') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)

# 分割标签和数据集
texts = [x[1] for x in text_data]
targets = [x[0] for x in text_data]
# 去除大小写产生的差异、去除标点、去除数字、去除多余的空格
texts = [x.lower() for x in texts]
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
texts = [' '.join(x.split()) for x in texts]


def tokenizer(text):
    return nltk.word_tokenize(text)


# 转化为tfidf
tfidf_texts = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features).fit_transform(texts)
# 随机分割数据集 产生80%训练集以及20%测试集
train_indics = np.random.choice(tfidf_texts.shape[0], round(0.8*tfidf_texts.shape[0]), replace=False)
test_indics = np.array(list(set(range(tfidf_texts.shape[0]))-set(train_indics)))
x_train = tfidf_texts[train_indics]
x_test = tfidf_texts[test_indics]
y_target = [1 if y == 'spam' else 0 for y in targets]
y_train = np.array([y for iy, y in enumerate(y_target) if iy in train_indics])
y_test = np.array([y for iy, y in enumerate(y_target) if iy in test_indics])

A = tf.Variable(tf.random_normal(shape=[max_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

model_output = tf.add(tf.matmul(x_data, A), b)

optimizer = tf.train.GradientDescentOptimizer(0.0025)
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_data))
train_step = optimizer.minimize(loss)

prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_data), dtype=tf.float32)
acc = tf.reduce_mean(predictions_correct)

init = tf.global_variables_initializer()
sess.run(init)
# 以批尺寸200训练10000个迭代
for i in range(10000):
    rand_indics = np.random.choice(x_train.shape[0], batch_size)
    rand_x = x_train[rand_indics].todense()
    rand_y = np.transpose([y_train[rand_indics]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y})
    if (i + 1) % 1000 == 0:
        print(i+1, ' acc:', sess.run(acc, feed_dict={x_data: x_test.todense(), y_data: np.transpose([y_test])}))

